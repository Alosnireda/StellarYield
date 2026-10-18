#![no_std]
#![allow(clippy::too_many_arguments)]

//! # Dutch Auction Liquidation Engine
//!
//! Implements a Continuous Dutch Auction mechanism for graceful liquidation
//! of undercollateralized vault positions.
//!
//! ## Problem
//! Fixed-discount liquidation dumps collateral at a static price, causing
//! massive price impact and MEV extraction. This hurts both vault owners
//! (worse price) and the protocol (higher bad debt risk).
//!
//! ## Solution — Continuous Dutch Auction
//! When a vault becomes undercollateralized:
//! 1. An auction starts at a high price (above market) for the collateral
//! 2. The price decays linearly over time toward a floor
//! 3. Any liquidator can buy collateral at the current auction price
//! 4. Early liquidators pay more, late ones get a discount
//! 5. Competition ensures fair price discovery near market value
//!
//! ## Benefits
//! - Minimal price impact (gradual price discovery)
//! - No fixed discount exploitation
//! - Incentivizes competitive liquidation
//! - Remaining collateral returned to vault owner
//!
//! ## Lifecycle
//! `Active` is the only state in which collateral can be bought, settled or
//! cancelled. `Completed` (fully filled or settled after expiry) and
//! `Cancelled` are terminal: no further fills or state transitions happen.
//!
//! ## Escrow & Withdrawals
//! Debt tokens paid by buyers are escrowed as seller proceeds and pulled by
//! the vault owner via `withdraw_proceeds` at any time after a fill. Unsold
//! collateral is pulled via `withdraw_unsold`, only once the auction has
//! reached a terminal state. Each balance can be withdrawn exactly once.

use soroban_sdk::{
    contract, contracterror, contractimpl, contracttype, symbol_short, token, Address, BytesN, Env,
};

mod math;

// ── Storage Keys ────────────────────────────────────────────────────────

#[contracttype]
#[derive(Clone, Debug)]
pub enum DataKey {
    Admin,
    Initialized,
    Auction(u64),
    NextAuctionId,
    VaultContract,
    OracleContract,
    // Global parameters
    MinCollateralRatio,    // e.g. 15000 = 150% (scaled by 10000)
    AuctionDuration,       // Duration in seconds for full price decay
    StartPremiumBps,       // Starting premium above oracle price in bps
    FloorDiscountBps,      // Floor discount below oracle price in bps
    LiquidationPenaltyBps, // Penalty kept by protocol in bps
    ProtocolFeeRecipient,
    /// Fill receipt keyed by a caller-supplied purchase identifier.
    Purchase(BytesN<32>),
}

// ── Data Structures ─────────────────────────────────────────────────────

#[contracttype]
#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum AuctionStatus {
    Active = 0,
    Completed = 1,
    Cancelled = 2,
}

impl AuctionStatus {
    /// `Completed` and `Cancelled` are terminal: no fills, settlement or
    /// cancellation may follow, only withdrawals by the seller.
    pub fn is_terminal(&self) -> bool {
        matches!(self, AuctionStatus::Completed | AuctionStatus::Cancelled)
    }
}

/// Recorded outcome of an identified purchase, used to make retries idempotent.
#[contracttype]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PurchaseReceipt {
    pub auction_id: u64,
    pub buyer: Address,
    pub requested_collateral: i128,
    pub max_debt_payment: i128,
    pub collateral_received: i128,
    pub debt_paid: i128,
}

#[contracttype]
#[derive(Clone, Debug)]
pub struct Auction {
    pub id: u64,
    pub vault_owner: Address,
    pub collateral_token: Address,
    pub debt_token: Address,
    pub collateral_amount: i128,
    pub debt_to_cover: i128,
    pub start_price: i128, // Starting price per unit of collateral (scaled 1e7)
    pub floor_price: i128, // Floor price per unit of collateral (scaled 1e7)
    pub start_time: u64,
    pub duration: u64,
    pub status: AuctionStatus,
    pub collateral_sold: i128,
    pub debt_recovered: i128,
    pub initiator: Address,
    /// Portion of `debt_recovered` already withdrawn by the seller.
    pub proceeds_withdrawn: i128,
    /// Whether unsold collateral has been withdrawn after a terminal state.
    pub unsold_withdrawn: bool,
}

// ── Errors ──────────────────────────────────────────────────────────────

#[contracterror]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum AuctionError {
    NotInitialized = 1,
    AlreadyInitialized = 2,
    ZeroAmount = 3,
    Unauthorized = 4,
    AuctionNotFound = 5,
    AuctionNotActive = 6,
    AuctionExpired = 7,
    VaultSufficient = 8,
    InsufficientCollateral = 9,
    InvalidParams = 10,
    AuctionStillActive = 11,
    NothingToLiquidate = 12,
    /// A purchase identifier was reused with a different payload.
    PurchaseIdConflict = 13,
    /// No proceeds or unsold collateral are currently withdrawable.
    NothingToWithdraw = 14,
    /// A fill would push sold collateral past the auctioned inventory.
    InventoryExceeded = 15,
}

/// Price precision: 1e7
const PRICE_SCALE: i128 = 10_000_000;
/// Ratio precision: 1e4 (basis points)
const BPS_SCALE: i128 = 10_000;

// ── Contract ────────────────────────────────────────────────────────────

#[contract]
pub struct DutchAuction;

#[allow(clippy::too_many_arguments)]
#[contractimpl]
impl DutchAuction {
    // ── Initialization ──────────────────────────────────────────────

    /// Initialize the Dutch Auction liquidation engine.
    ///
    /// # Arguments
    /// * `admin` — Protocol admin
    /// * `vault_contract` — Address of the vault contract to monitor
    /// * `oracle_contract` — Price oracle address
    /// * `min_collateral_ratio` — Min ratio in bps (e.g. 15000 = 150%)
    /// * `auction_duration` — Auction duration in seconds
    /// * `start_premium_bps` — Starting price premium above oracle (e.g. 2000 = 20%)
    /// * `floor_discount_bps` — Floor discount below oracle (e.g. 1000 = 10%)
    /// * `penalty_bps` — Liquidation penalty kept by protocol
    /// * `fee_recipient` — Protocol fee recipient
    pub fn initialize(
        env: Env,
        admin: Address,
        vault_contract: Address,
        oracle_contract: Address,
        min_collateral_ratio: u32,
        auction_duration: u64,
        start_premium_bps: u32,
        floor_discount_bps: u32,
        penalty_bps: u32,
        fee_recipient: Address,
    ) -> Result<(), AuctionError> {
        if env.storage().instance().has(&DataKey::Initialized) {
            return Err(AuctionError::AlreadyInitialized);
        }

        env.storage().instance().set(&DataKey::Admin, &admin);
        env.storage()
            .instance()
            .set(&DataKey::VaultContract, &vault_contract);
        env.storage()
            .instance()
            .set(&DataKey::OracleContract, &oracle_contract);
        env.storage()
            .instance()
            .set(&DataKey::MinCollateralRatio, &min_collateral_ratio);
        env.storage()
            .instance()
            .set(&DataKey::AuctionDuration, &auction_duration);
        env.storage()
            .instance()
            .set(&DataKey::StartPremiumBps, &start_premium_bps);
        env.storage()
            .instance()
            .set(&DataKey::FloorDiscountBps, &floor_discount_bps);
        env.storage()
            .instance()
            .set(&DataKey::LiquidationPenaltyBps, &penalty_bps);
        env.storage()
            .instance()
            .set(&DataKey::ProtocolFeeRecipient, &fee_recipient);
        env.storage().instance().set(&DataKey::NextAuctionId, &1u64);
        env.storage().instance().set(&DataKey::Initialized, &true);

        env.events().publish(
            (symbol_short!("init"),),
            (admin, min_collateral_ratio, auction_duration),
        );

        Ok(())
    }

    // ── Start Auction ───────────────────────────────────────────────

    /// Start a Dutch auction for an undercollateralized vault position.
    ///
    /// Anyone can call this to initiate liquidation of an unhealthy position.
    /// The auction starts at a premium above oracle price and decays to a
    /// floor discount over the configured duration.
    ///
    /// # Arguments
    /// * `initiator` — Address starting the auction (receives incentive)
    /// * `vault_owner` — Owner of the undercollateralized vault
    /// * `collateral_token` — The collateral token address
    /// * `debt_token` — The debt token address
    /// * `collateral_amount` — Amount of collateral to auction
    /// * `debt_to_cover` — Amount of debt that must be repaid
    /// * `oracle_price` — Current oracle price (scaled 1e7)
    pub fn start_auction(
        env: Env,
        initiator: Address,
        vault_owner: Address,
        collateral_token: Address,
        debt_token: Address,
        collateral_amount: i128,
        debt_to_cover: i128,
        oracle_price: i128,
    ) -> Result<u64, AuctionError> {
        Self::require_init(&env)?;
        initiator.require_auth();
        vault_owner.require_auth();

        if collateral_amount <= 0 || debt_to_cover <= 0 || oracle_price <= 0 {
            return Err(AuctionError::ZeroAmount);
        }

        // Verify the position is undercollateralized
        let min_ratio: u32 = env
            .storage()
            .instance()
            .get(&DataKey::MinCollateralRatio)
            .unwrap();
        let collateral_value = (collateral_amount * oracle_price) / PRICE_SCALE;
        let required_collateral = (debt_to_cover * min_ratio as i128) / BPS_SCALE;

        if collateral_value >= required_collateral {
            return Err(AuctionError::VaultSufficient);
        }

        // Calculate start and floor prices
        let start_premium: u32 = env
            .storage()
            .instance()
            .get(&DataKey::StartPremiumBps)
            .unwrap();
        let floor_discount: u32 = env
            .storage()
            .instance()
            .get(&DataKey::FloorDiscountBps)
            .unwrap();

        let start_price = oracle_price + (oracle_price * start_premium as i128) / BPS_SCALE;
        let floor_price = oracle_price - (oracle_price * floor_discount as i128) / BPS_SCALE;

        let duration: u64 = env
            .storage()
            .instance()
            .get(&DataKey::AuctionDuration)
            .unwrap();

        // Transfer collateral from vault owner to auction contract
        let collateral_client = token::Client::new(&env, &collateral_token);
        collateral_client.transfer(
            &vault_owner,
            &env.current_contract_address(),
            &collateral_amount,
        );

        let auction_id: u64 = env
            .storage()
            .instance()
            .get(&DataKey::NextAuctionId)
            .unwrap();

        let now = env.ledger().timestamp();
        let auction = Auction {
            id: auction_id,
            vault_owner: vault_owner.clone(),
            collateral_token: collateral_token.clone(),
            debt_token: debt_token.clone(),
            collateral_amount,
            debt_to_cover,
            start_price,
            floor_price,
            start_time: now,
            duration,
            status: AuctionStatus::Active,
            collateral_sold: 0,
            debt_recovered: 0,
            initiator: initiator.clone(),
            proceeds_withdrawn: 0,
            unsold_withdrawn: false,
        };

        env.storage()
            .persistent()
            .set(&DataKey::Auction(auction_id), &auction);
        env.storage()
            .instance()
            .set(&DataKey::NextAuctionId, &(auction_id + 1));

        env.events().publish(
            (symbol_short!("auction"),),
            (auction_id, vault_owner, collateral_amount, debt_to_cover),
        );

        Ok(auction_id)
    }

    // ── Buy Collateral ──────────────────────────────────────────────

    /// Buy collateral from an active Dutch auction at the current price.
    ///
    /// The price starts high and decays linearly toward the floor.
    /// The liquidator pays debt tokens (escrowed as seller proceeds) and
    /// receives collateral. Requests beyond the remaining inventory are
    /// capped, so total sold collateral never exceeds the auctioned amount.
    ///
    /// # Arguments
    /// * `buyer` — Liquidator address (must authorize)
    /// * `auction_id` — Active auction to buy from
    /// * `collateral_amount` — Amount of collateral to purchase
    /// * `max_debt_payment` — Maximum debt tokens willing to pay (slippage protection)
    ///
    /// # Returns
    /// (collateral_received, debt_paid) — Actual amounts exchanged
    pub fn buy_collateral(
        env: Env,
        buyer: Address,
        auction_id: u64,
        collateral_amount: i128,
        max_debt_payment: i128,
    ) -> Result<(i128, i128), AuctionError> {
        Self::require_init(&env)?;
        buyer.require_auth();

        Self::execute_fill(
            &env,
            &buyer,
            auction_id,
            collateral_amount,
            max_debt_payment,
        )
    }

    /// Idempotent variant of `buy_collateral` keyed by `purchase_id`.
    ///
    /// The first call executes the fill and records a receipt. A retry with
    /// the same identifier and identical payload returns the original
    /// receipt amounts without moving tokens or touching fill accounting.
    /// A retry with the same identifier but a different payload is rejected
    /// with `PurchaseIdConflict`.
    pub fn buy_collateral_with_id(
        env: Env,
        buyer: Address,
        auction_id: u64,
        purchase_id: BytesN<32>,
        collateral_amount: i128,
        max_debt_payment: i128,
    ) -> Result<(i128, i128), AuctionError> {
        Self::require_init(&env)?;
        buyer.require_auth();

        let key = DataKey::Purchase(purchase_id.clone());
        if let Some(receipt) = env.storage().persistent().get::<_, PurchaseReceipt>(&key) {
            if receipt.auction_id != auction_id
                || receipt.buyer != buyer
                || receipt.requested_collateral != collateral_amount
                || receipt.max_debt_payment != max_debt_payment
            {
                return Err(AuctionError::PurchaseIdConflict);
            }
            env.events()
                .publish((symbol_short!("replay"),), (auction_id, purchase_id));
            return Ok((receipt.collateral_received, receipt.debt_paid));
        }

        let (collateral_received, debt_paid) = Self::execute_fill(
            &env,
            &buyer,
            auction_id,
            collateral_amount,
            max_debt_payment,
        )?;

        env.storage().persistent().set(
            &key,
            &PurchaseReceipt {
                auction_id,
                buyer,
                requested_collateral: collateral_amount,
                max_debt_payment,
                collateral_received,
                debt_paid,
            },
        );

        Ok((collateral_received, debt_paid))
    }

    // ── Settle Expired Auction ──────────────────────────────────────

    /// Settle an auction that has passed its duration without being
    /// fully filled. Terminal auctions are rejected before any state change;
    /// remaining collateral becomes withdrawable by the vault owner.
    pub fn settle_auction(env: Env, auction_id: u64) -> Result<(), AuctionError> {
        Self::require_init(&env)?;

        let mut auction = Self::load_active(&env, auction_id)?;

        let now = env.ledger().timestamp();
        if now < auction.start_time + auction.duration {
            return Err(AuctionError::AuctionStillActive);
        }

        auction.status = AuctionStatus::Completed;
        let leftover = Self::remaining_inventory(&auction)?;

        env.storage()
            .persistent()
            .set(&DataKey::Auction(auction_id), &auction);

        env.events()
            .publish((symbol_short!("settle"),), (auction_id, leftover));

        Ok(())
    }

    // ── Seller Withdrawals ──────────────────────────────────────────

    /// Withdraw escrowed debt-token proceeds. Vault owner only.
    ///
    /// Proceeds are final once a fill executes, so they can be withdrawn at
    /// any time; only the not-yet-withdrawn balance is paid out.
    pub fn withdraw_proceeds(
        env: Env,
        seller: Address,
        auction_id: u64,
    ) -> Result<i128, AuctionError> {
        Self::require_init(&env)?;
        seller.require_auth();

        let mut auction = Self::load_auction(&env, auction_id)?;
        if seller != auction.vault_owner {
            return Err(AuctionError::Unauthorized);
        }

        let owed = auction.debt_recovered - auction.proceeds_withdrawn;
        if owed <= 0 {
            return Err(AuctionError::NothingToWithdraw);
        }

        auction.proceeds_withdrawn = auction.debt_recovered;
        env.storage()
            .persistent()
            .set(&DataKey::Auction(auction_id), &auction);

        token::Client::new(&env, &auction.debt_token).transfer(
            &env.current_contract_address(),
            &seller,
            &owed,
        );

        env.events()
            .publish((symbol_short!("wd_proc"),), (auction_id, seller, owed));

        Ok(owed)
    }

    /// Withdraw collateral left unsold. Vault owner only, and only once the
    /// auction is terminal — while active the inventory is still for sale.
    pub fn withdraw_unsold(
        env: Env,
        seller: Address,
        auction_id: u64,
    ) -> Result<i128, AuctionError> {
        Self::require_init(&env)?;
        seller.require_auth();

        let mut auction = Self::load_auction(&env, auction_id)?;
        if seller != auction.vault_owner {
            return Err(AuctionError::Unauthorized);
        }
        if !auction.status.is_terminal() {
            return Err(AuctionError::AuctionStillActive);
        }

        let leftover = Self::remaining_inventory(&auction)?;
        if auction.unsold_withdrawn || leftover <= 0 {
            return Err(AuctionError::NothingToWithdraw);
        }

        auction.unsold_withdrawn = true;
        env.storage()
            .persistent()
            .set(&DataKey::Auction(auction_id), &auction);

        token::Client::new(&env, &auction.collateral_token).transfer(
            &env.current_contract_address(),
            &seller,
            &leftover,
        );

        env.events().publish(
            (symbol_short!("wd_unsold"),),
            (auction_id, seller, leftover),
        );

        Ok(leftover)
    }

    // ── Admin ───────────────────────────────────────────────────────

    /// Cancel an active auction. Admin only. Emergency use.
    /// Remaining collateral becomes withdrawable by the vault owner.
    pub fn cancel_auction(env: Env, admin: Address, auction_id: u64) -> Result<(), AuctionError> {
        Self::require_admin(&env, &admin)?;

        let mut auction = Self::load_active(&env, auction_id)?;
        auction.status = AuctionStatus::Cancelled;

        env.storage()
            .persistent()
            .set(&DataKey::Auction(auction_id), &auction);

        env.events()
            .publish((symbol_short!("cancel"),), (auction_id,));

        Ok(())
    }

    /// Update auction parameters. Admin only.
    pub fn update_params(
        env: Env,
        admin: Address,
        min_collateral_ratio: u32,
        auction_duration: u64,
        start_premium_bps: u32,
        floor_discount_bps: u32,
        penalty_bps: u32,
    ) -> Result<(), AuctionError> {
        Self::require_admin(&env, &admin)?;

        env.storage()
            .instance()
            .set(&DataKey::MinCollateralRatio, &min_collateral_ratio);
        env.storage()
            .instance()
            .set(&DataKey::AuctionDuration, &auction_duration);
        env.storage()
            .instance()
            .set(&DataKey::StartPremiumBps, &start_premium_bps);
        env.storage()
            .instance()
            .set(&DataKey::FloorDiscountBps, &floor_discount_bps);
        env.storage()
            .instance()
            .set(&DataKey::LiquidationPenaltyBps, &penalty_bps);

        env.events().publish(
            (symbol_short!("params"),),
            (min_collateral_ratio, auction_duration),
        );

        Ok(())
    }

    // ── View Functions ──────────────────────────────────────────────

    pub fn get_auction(env: Env, auction_id: u64) -> Result<Auction, AuctionError> {
        env.storage()
            .persistent()
            .get(&DataKey::Auction(auction_id))
            .ok_or(AuctionError::AuctionNotFound)
    }

    /// Get the current price for an active auction.
    /// Returns the linearly-decayed price at the current timestamp.
    pub fn get_current_price(env: Env, auction_id: u64) -> Result<i128, AuctionError> {
        let auction: Auction = env
            .storage()
            .persistent()
            .get(&DataKey::Auction(auction_id))
            .ok_or(AuctionError::AuctionNotFound)?;

        if auction.status != AuctionStatus::Active {
            return Err(AuctionError::AuctionNotActive);
        }

        let now = env.ledger().timestamp();
        Ok(math::get_current_price(
            auction.start_price,
            auction.floor_price,
            auction.start_time,
            auction.duration,
            now,
        ))
    }

    /// Calculate the cost to buy a given amount of collateral at the current price.
    pub fn quote_buy(
        env: Env,
        auction_id: u64,
        collateral_amount: i128,
    ) -> Result<i128, AuctionError> {
        let auction: Auction = env
            .storage()
            .persistent()
            .get(&DataKey::Auction(auction_id))
            .ok_or(AuctionError::AuctionNotFound)?;

        if auction.status != AuctionStatus::Active {
            return Err(AuctionError::AuctionNotActive);
        }

        let now = env.ledger().timestamp();
        let price = math::get_current_price(
            auction.start_price,
            auction.floor_price,
            auction.start_time,
            auction.duration,
            now,
        );

        let remaining = auction.collateral_amount - auction.collateral_sold;
        let actual = if collateral_amount > remaining {
            remaining
        } else {
            collateral_amount
        };

        let penalty_bps: u32 = env
            .storage()
            .instance()
            .get(&DataKey::LiquidationPenaltyBps)
            .unwrap_or(0);

        let base_cost = (actual * price) / PRICE_SCALE;
        let penalty = (base_cost * penalty_bps as i128) / BPS_SCALE;

        Ok(base_cost + penalty)
    }

    pub fn get_min_collateral_ratio(env: Env) -> u32 {
        env.storage()
            .instance()
            .get(&DataKey::MinCollateralRatio)
            .unwrap_or(15000)
    }

    pub fn get_auction_duration(env: Env) -> u64 {
        env.storage()
            .instance()
            .get(&DataKey::AuctionDuration)
            .unwrap_or(3600)
    }

    /// Collateral still available to buyers (or, once terminal, unsold).
    pub fn get_remaining_inventory(env: Env, auction_id: u64) -> Result<i128, AuctionError> {
        let auction = Self::load_auction(&env, auction_id)?;
        Self::remaining_inventory(&auction)
    }

    /// Amounts the vault owner can withdraw right now:
    /// (debt-token proceeds, unsold collateral).
    pub fn get_withdrawable(env: Env, auction_id: u64) -> Result<(i128, i128), AuctionError> {
        let auction = Self::load_auction(&env, auction_id)?;
        let proceeds = auction.debt_recovered - auction.proceeds_withdrawn;
        let unsold = if auction.status.is_terminal() && !auction.unsold_withdrawn {
            Self::remaining_inventory(&auction)?
        } else {
            0
        };
        Ok((proceeds, unsold))
    }

    pub fn get_purchase(env: Env, purchase_id: BytesN<32>) -> Option<PurchaseReceipt> {
        env.storage()
            .persistent()
            .get(&DataKey::Purchase(purchase_id))
    }

    // ── Internal ────────────────────────────────────────────────────

    fn load_auction(env: &Env, auction_id: u64) -> Result<Auction, AuctionError> {
        env.storage()
            .persistent()
            .get(&DataKey::Auction(auction_id))
            .ok_or(AuctionError::AuctionNotFound)
    }

    /// Load an auction and reject terminal states before any side effect.
    fn load_active(env: &Env, auction_id: u64) -> Result<Auction, AuctionError> {
        let auction = Self::load_auction(env, auction_id)?;
        if auction.status.is_terminal() {
            return Err(AuctionError::AuctionNotActive);
        }
        Ok(auction)
    }

    fn remaining_inventory(auction: &Auction) -> Result<i128, AuctionError> {
        let remaining = auction
            .collateral_amount
            .checked_sub(auction.collateral_sold)
            .ok_or(AuctionError::InventoryExceeded)?;
        if remaining < 0 {
            return Err(AuctionError::InventoryExceeded);
        }
        Ok(remaining)
    }

    /// Execute a fill. Accounting is committed before any token movement so
    /// the inventory bound holds across every buyer.
    fn execute_fill(
        env: &Env,
        buyer: &Address,
        auction_id: u64,
        collateral_amount: i128,
        max_debt_payment: i128,
    ) -> Result<(i128, i128), AuctionError> {
        if collateral_amount <= 0 {
            return Err(AuctionError::ZeroAmount);
        }

        let mut auction = Self::load_active(env, auction_id)?;

        let remaining_collateral = Self::remaining_inventory(&auction)?;
        if remaining_collateral <= 0 {
            return Err(AuctionError::InsufficientCollateral);
        }

        // Cap at remaining collateral
        let actual_collateral = collateral_amount.min(remaining_collateral);

        // Get current auction price (decays linearly)
        let now = env.ledger().timestamp();
        let current_price = math::get_current_price(
            auction.start_price,
            auction.floor_price,
            auction.start_time,
            auction.duration,
            now,
        );

        // Calculate debt payment = collateral * current_price / PRICE_SCALE
        let debt_payment = actual_collateral
            .checked_mul(current_price)
            .ok_or(AuctionError::InvalidParams)?
            / PRICE_SCALE;
        if debt_payment <= 0 {
            return Err(AuctionError::ZeroAmount);
        }

        if debt_payment > max_debt_payment {
            return Err(AuctionError::InvalidParams);
        }

        // Cap debt recovery at remaining debt
        let remaining_debt = auction.debt_to_cover - auction.debt_recovered;
        let actual_debt = debt_payment.min(remaining_debt);

        // Calculate liquidation penalty (protocol fee)
        let penalty_bps: u32 = env
            .storage()
            .instance()
            .get(&DataKey::LiquidationPenaltyBps)
            .unwrap_or(0);
        let penalty = (actual_debt * penalty_bps as i128) / BPS_SCALE;
        let total_from_buyer = actual_debt + penalty;

        // Update auction state
        let new_sold = auction
            .collateral_sold
            .checked_add(actual_collateral)
            .ok_or(AuctionError::InventoryExceeded)?;
        if new_sold > auction.collateral_amount {
            return Err(AuctionError::InventoryExceeded);
        }
        auction.collateral_sold = new_sold;
        auction.debt_recovered += actual_debt;

        // Complete auction if all debt covered or all collateral sold.
        // Any leftover collateral stays escrowed for `withdraw_unsold`.
        if auction.debt_recovered >= auction.debt_to_cover
            || auction.collateral_sold >= auction.collateral_amount
        {
            auction.status = AuctionStatus::Completed;
        }

        env.storage()
            .persistent()
            .set(&DataKey::Auction(auction_id), &auction);

        // Debt payment is escrowed as seller proceeds
        let contract = env.current_contract_address();
        let debt_client = token::Client::new(env, &auction.debt_token);
        debt_client.transfer(buyer, &contract, &actual_debt);

        // Penalty goes to protocol
        if penalty > 0 {
            let fee_recipient: Address = env
                .storage()
                .instance()
                .get(&DataKey::ProtocolFeeRecipient)
                .unwrap();
            debt_client.transfer(buyer, &fee_recipient, &penalty);
        }

        // Transfer collateral from contract to buyer
        token::Client::new(env, &auction.collateral_token).transfer(
            &contract,
            buyer,
            &actual_collateral,
        );

        env.events().publish(
            (symbol_short!("buy"),),
            (
                auction_id,
                buyer.clone(),
                actual_collateral,
                total_from_buyer,
            ),
        );

        Ok((actual_collateral, total_from_buyer))
    }

    fn require_init(env: &Env) -> Result<(), AuctionError> {
        if !env.storage().instance().has(&DataKey::Initialized) {
            return Err(AuctionError::NotInitialized);
        }
        Ok(())
    }

    fn require_admin(env: &Env, caller: &Address) -> Result<(), AuctionError> {
        Self::require_init(env)?;
        caller.require_auth();
        let admin: Address = env
            .storage()
            .instance()
            .get(&DataKey::Admin)
            .ok_or(AuctionError::NotInitialized)?;
        if *caller != admin {
            return Err(AuctionError::Unauthorized);
        }
        Ok(())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use soroban_sdk::testutils::{Address as _, Ledger};
    use soroban_sdk::Env;

    /// Setup helper:
    /// - min_collateral_ratio = 15000 (150%)
    /// - auction_duration = 3600s (1 hour)
    /// - start_premium = 2000bps (20% above oracle)
    /// - floor_discount = 1000bps (10% below oracle)
    /// - penalty = 500bps (5%)
    fn setup_env() -> (
        Env,
        DutchAuctionClient<'static>,
        Address,
        Address,
        Address,
        Address,
    ) {
        let env = Env::default();
        env.mock_all_auths();

        let contract_id = env.register(DutchAuction, ());
        let client = DutchAuctionClient::new(&env, &contract_id);

        let admin = Address::generate(&env);
        let vault_contract = Address::generate(&env);
        let oracle = Address::generate(&env);
        let fee_recipient = Address::generate(&env);

        let collateral_contract = env.register_stellar_asset_contract_v2(admin.clone());
        let debt_contract = env.register_stellar_asset_contract_v2(fee_recipient.clone());
        let collateral_token = collateral_contract.address();
        let debt_token = debt_contract.address();

        client.initialize(
            &admin,
            &vault_contract,
            &oracle,
            &15000, // 150% min collateral ratio
            &3600,  // 1 hour auction duration
            &2000,  // 20% start premium
            &1000,  // 10% floor discount
            &500,   // 5% penalty
            &fee_recipient,
        );

        (
            env,
            client,
            admin,
            fee_recipient,
            collateral_token,
            debt_token,
        )
    }

    fn mint_tokens(env: &Env, token_addr: &Address, to: &Address, amount: i128) {
        let admin_client = soroban_sdk::token::StellarAssetClient::new(env, token_addr);
        admin_client.mint(to, &amount);
    }

    #[test]
    fn test_initialize() {
        let (_, client, _, _, _, _) = setup_env();
        assert_eq!(client.get_min_collateral_ratio(), 15000);
        assert_eq!(client.get_auction_duration(), 3600);
    }

    #[test]
    #[should_panic(expected = "Error(Contract, #2)")]
    fn test_double_initialize_panics() {
        let (_, client, admin, fee_recipient, _, _) = setup_env();
        let vault = Address::generate(client.address.env());
        let oracle = Address::generate(client.address.env());
        client.initialize(
            &admin,
            &vault,
            &oracle,
            &15000,
            &3600,
            &2000,
            &1000,
            &500,
            &fee_recipient,
        );
    }

    #[test]
    fn test_start_auction() {
        let (env, client, _, _, collateral_token, debt_token) = setup_env();

        let vault_owner = Address::generate(&env);
        let initiator = Address::generate(&env);

        // Vault owner has 1000 collateral, 800 debt
        // At oracle price 1.0 (1e7), collateral value = 1000, required = 800 * 1.5 = 1200
        // So the vault IS undercollateralized
        mint_tokens(&env, &collateral_token, &vault_owner, 1_000);

        env.ledger().set_timestamp(1000);

        let oracle_price: i128 = 10_000_000; // 1.0 scaled by 1e7
        let auction_id = client.start_auction(
            &initiator,
            &vault_owner,
            &collateral_token,
            &debt_token,
            &1_000,
            &800,
            &oracle_price,
        );

        assert_eq!(auction_id, 1);
        let auction = client.get_auction(&auction_id);
        assert_eq!(auction.status, AuctionStatus::Active);
        assert_eq!(auction.collateral_amount, 1_000);
        assert_eq!(auction.debt_to_cover, 800);

        // Start price = oracle + 20% = 12_000_000
        assert_eq!(auction.start_price, 12_000_000);
        // Floor price = oracle - 10% = 9_000_000
        assert_eq!(auction.floor_price, 9_000_000);
    }

    #[test]
    #[should_panic(expected = "Error(Contract, #8)")]
    fn test_start_auction_vault_sufficient_panics() {
        let (env, client, _, _, collateral_token, debt_token) = setup_env();

        let vault_owner = Address::generate(&env);
        let initiator = Address::generate(&env);

        // Vault owner has 2000 collateral, 800 debt
        // collateral value = 2000, required = 800 * 1.5 = 1200 → sufficient
        mint_tokens(&env, &collateral_token, &vault_owner, 2_000);

        env.ledger().set_timestamp(1000);

        let oracle_price: i128 = 10_000_000;
        client.start_auction(
            &initiator,
            &vault_owner,
            &collateral_token,
            &debt_token,
            &2_000,
            &800,
            &oracle_price,
        );
    }

    #[test]
    fn test_price_decay() {
        let (env, client, _, _, collateral_token, debt_token) = setup_env();

        let vault_owner = Address::generate(&env);
        let initiator = Address::generate(&env);

        mint_tokens(&env, &collateral_token, &vault_owner, 1_000);
        env.ledger().set_timestamp(1000);

        let oracle_price: i128 = 10_000_000;
        let auction_id = client.start_auction(
            &initiator,
            &vault_owner,
            &collateral_token,
            &debt_token,
            &1_000,
            &800,
            &oracle_price,
        );

        // At start: price = 12_000_000 (oracle + 20%)
        let price_at_start = client.get_current_price(&auction_id);
        assert_eq!(price_at_start, 12_000_000);

        // At halfway (1800s): price = 12M - (3M * 1800/3600) = 12M - 1.5M = 10_500_000
        env.ledger().set_timestamp(2800);
        let price_halfway = client.get_current_price(&auction_id);
        assert_eq!(price_halfway, 10_500_000);

        // At end (3600s): price = floor = 9_000_000
        env.ledger().set_timestamp(4600);
        let price_at_end = client.get_current_price(&auction_id);
        assert_eq!(price_at_end, 9_000_000);

        // Past end: still floor
        env.ledger().set_timestamp(10000);
        let price_past = client.get_current_price(&auction_id);
        assert_eq!(price_past, 9_000_000);
    }

    #[test]
    fn test_buy_collateral() {
        let (env, client, _, fee_recipient, collateral_token, debt_token) = setup_env();

        let vault_owner = Address::generate(&env);
        let initiator = Address::generate(&env);
        let liquidator = Address::generate(&env);

        mint_tokens(&env, &collateral_token, &vault_owner, 1_000);
        mint_tokens(&env, &debt_token, &liquidator, 50_000);

        env.ledger().set_timestamp(1000);

        let oracle_price: i128 = 10_000_000;
        let auction_id = client.start_auction(
            &initiator,
            &vault_owner,
            &collateral_token,
            &debt_token,
            &1_000,
            &800,
            &oracle_price,
        );

        // Buy 500 collateral at halfway point
        env.ledger().set_timestamp(2800);
        // Price at halfway = 10_500_000
        // Cost = 500 * 10_500_000 / 10_000_000 = 525
        // Penalty = 525 * 500 / 10000 = 26 (5%)
        // Total from buyer = 525 + 26 = 551

        let (coll_received, debt_paid) = client.buy_collateral(
            &liquidator,
            &auction_id,
            &500,
            &600, // max willing to pay
        );

        assert_eq!(coll_received, 500);
        assert_eq!(debt_paid, 551); // 525 debt + 26 penalty

        // Verify balances
        let coll_client = token::Client::new(&env, &collateral_token);
        assert_eq!(coll_client.balance(&liquidator), 500);

        let debt_client = token::Client::new(&env, &debt_token);
        // 525 (debt repayment) is escrowed until the vault owner withdraws it
        assert_eq!(debt_client.balance(&client.address), 525);
        assert_eq!(debt_client.balance(&vault_owner), 0);
        // Fee recipient received 26 (penalty)
        assert_eq!(debt_client.balance(&fee_recipient), 26);

        assert_eq!(client.withdraw_proceeds(&vault_owner, &auction_id), 525);
        assert_eq!(debt_client.balance(&vault_owner), 525);

        // Auction still active (500 collateral remaining)
        let auction = client.get_auction(&auction_id);
        assert_eq!(auction.status, AuctionStatus::Active);
        assert_eq!(auction.collateral_sold, 500);
        assert_eq!(auction.debt_recovered, 525);
    }

    #[test]
    fn test_buy_all_collateral_completes_auction() {
        let (env, client, _, _, collateral_token, debt_token) = setup_env();

        let vault_owner = Address::generate(&env);
        let initiator = Address::generate(&env);
        let liquidator = Address::generate(&env);

        mint_tokens(&env, &collateral_token, &vault_owner, 1_000);
        mint_tokens(&env, &debt_token, &liquidator, 50_000);

        env.ledger().set_timestamp(1000);

        let oracle_price: i128 = 10_000_000;
        let auction_id = client.start_auction(
            &initiator,
            &vault_owner,
            &collateral_token,
            &debt_token,
            &1_000,
            &800,
            &oracle_price,
        );

        // Buy ALL collateral at start price
        env.ledger().set_timestamp(1000);
        // Price = 12_000_000
        // Cost = 1000 * 12M / 10M = 1200 → capped at remaining debt 800
        // Penalty = 800 * 500 / 10000 = 40
        // Actually, cost is 1200 which > debt_to_cover (800), so debt_paid = 800
        // Total = 800 + 40 = 840

        let (coll_received, _debt_paid) = client.buy_collateral(
            &liquidator,
            &auction_id,
            &1_000,
            &2_000, // max willing
        );

        assert_eq!(coll_received, 1_000);

        // Auction completed
        let auction = client.get_auction(&auction_id);
        assert_eq!(auction.status, AuctionStatus::Completed);
    }

    #[test]
    fn test_settle_expired_auction() {
        let (env, client, _, _, collateral_token, debt_token) = setup_env();

        let vault_owner = Address::generate(&env);
        let initiator = Address::generate(&env);

        mint_tokens(&env, &collateral_token, &vault_owner, 1_000);

        env.ledger().set_timestamp(1000);

        let oracle_price: i128 = 10_000_000;
        let auction_id = client.start_auction(
            &initiator,
            &vault_owner,
            &collateral_token,
            &debt_token,
            &1_000,
            &800,
            &oracle_price,
        );

        // Advance past auction duration
        env.ledger().set_timestamp(5000);

        client.settle_auction(&auction_id);

        let auction = client.get_auction(&auction_id);
        assert_eq!(auction.status, AuctionStatus::Completed);

        // All collateral withdrawable by vault owner
        assert_eq!(client.withdraw_unsold(&vault_owner, &auction_id), 1_000);
        let coll_client = token::Client::new(&env, &collateral_token);
        assert_eq!(coll_client.balance(&vault_owner), 1_000);
    }

    #[test]
    fn test_cancel_auction_admin() {
        let (env, client, admin, _, collateral_token, debt_token) = setup_env();

        let vault_owner = Address::generate(&env);
        let initiator = Address::generate(&env);

        mint_tokens(&env, &collateral_token, &vault_owner, 1_000);

        env.ledger().set_timestamp(1000);

        let oracle_price: i128 = 10_000_000;
        let auction_id = client.start_auction(
            &initiator,
            &vault_owner,
            &collateral_token,
            &debt_token,
            &1_000,
            &800,
            &oracle_price,
        );

        client.cancel_auction(&admin, &auction_id);

        let auction = client.get_auction(&auction_id);
        assert_eq!(auction.status, AuctionStatus::Cancelled);

        // Collateral returned
        assert_eq!(client.withdraw_unsold(&vault_owner, &auction_id), 1_000);
        let coll_client = token::Client::new(&env, &collateral_token);
        assert_eq!(coll_client.balance(&vault_owner), 1_000);
    }

    #[test]
    fn test_quote_buy() {
        let (env, client, _, _, collateral_token, debt_token) = setup_env();

        let vault_owner = Address::generate(&env);
        let initiator = Address::generate(&env);

        mint_tokens(&env, &collateral_token, &vault_owner, 1_000);
        env.ledger().set_timestamp(1000);

        let oracle_price: i128 = 10_000_000;
        let auction_id = client.start_auction(
            &initiator,
            &vault_owner,
            &collateral_token,
            &debt_token,
            &1_000,
            &800,
            &oracle_price,
        );

        // Quote at start: price = 12M, buy 100 collateral
        // base_cost = 100 * 12M / 10M = 120
        // penalty = 120 * 500 / 10000 = 6
        // total = 126
        let quote = client.quote_buy(&auction_id, &100);
        assert_eq!(quote, 126);
    }

    #[test]
    #[should_panic(expected = "Error(Contract, #11)")]
    fn test_settle_before_expiry_panics() {
        let (env, client, _, _, collateral_token, debt_token) = setup_env();

        let vault_owner = Address::generate(&env);
        let initiator = Address::generate(&env);

        mint_tokens(&env, &collateral_token, &vault_owner, 1_000);
        env.ledger().set_timestamp(1000);

        let oracle_price: i128 = 10_000_000;
        let auction_id = client.start_auction(
            &initiator,
            &vault_owner,
            &collateral_token,
            &debt_token,
            &1_000,
            &800,
            &oracle_price,
        );

        // Try to settle before expiry
        env.ledger().set_timestamp(2000);
        client.settle_auction(&auction_id);
    }

    // ── Shared helpers for fill / lifecycle / withdrawal tests ──────

    /// Start an auction at t=1000 with oracle price 1.0 (start price 1.2,
    /// floor 0.9). Returns (auction_id, vault_owner).
    fn start_auction_with(
        env: &Env,
        client: &DutchAuctionClient,
        collateral_token: &Address,
        debt_token: &Address,
        collateral: i128,
        debt: i128,
    ) -> (u64, Address) {
        let vault_owner = Address::generate(env);
        let initiator = Address::generate(env);
        mint_tokens(env, collateral_token, &vault_owner, collateral);
        env.ledger().set_timestamp(1000);
        let id = client.start_auction(
            &initiator,
            &vault_owner,
            collateral_token,
            debt_token,
            &collateral,
            &debt,
            &10_000_000,
        );
        (id, vault_owner)
    }

    fn funded_buyer(env: &Env, debt_token: &Address) -> Address {
        let buyer = Address::generate(env);
        mint_tokens(env, debt_token, &buyer, 100_000);
        buyer
    }

    fn pid(env: &Env, byte: u8) -> BytesN<32> {
        BytesN::from_array(env, &[byte; 32])
    }

    // ── #1471: Idempotent fills by purchase identifier ──────────────

    #[test]
    fn test_purchase_id_identical_retry_is_idempotent() {
        let (env, client, _, fee_recipient, coll, debt) = setup_env();
        let (id, _) = start_auction_with(&env, &client, &coll, &debt, 1_000, 800);
        let buyer = funded_buyer(&env, &debt);
        let purchase = pid(&env, 1);

        // 300 @ 1.2 = 360 debt + 18 penalty
        let first = client.buy_collateral_with_id(&buyer, &id, &purchase, &300, &400);
        assert_eq!(first, (300, 378));

        let coll_client = token::Client::new(&env, &coll);
        let debt_client = token::Client::new(&env, &debt);
        let buyer_debt_before = debt_client.balance(&buyer);

        // Retry later, when the price has decayed — original receipt is returned
        env.ledger().set_timestamp(2800);
        let retry = client.buy_collateral_with_id(&buyer, &id, &purchase, &300, &400);
        assert_eq!(retry, first);

        // No additional token movement or fill accounting
        assert_eq!(coll_client.balance(&buyer), 300);
        assert_eq!(debt_client.balance(&buyer), buyer_debt_before);
        assert_eq!(debt_client.balance(&fee_recipient), 18);
        let auction = client.get_auction(&id);
        assert_eq!(auction.collateral_sold, 300);
        assert_eq!(auction.debt_recovered, 360);

        let receipt = client.get_purchase(&purchase).unwrap();
        assert_eq!(receipt.collateral_received, 300);
        assert_eq!(receipt.debt_paid, 378);
        assert_eq!(receipt.buyer, buyer);
    }

    #[test]
    fn test_purchase_id_retry_after_completion_returns_receipt() {
        let (env, client, _, _, coll, debt) = setup_env();
        let (id, _) = start_auction_with(&env, &client, &coll, &debt, 1_000, 800);
        let buyer = funded_buyer(&env, &debt);
        let purchase = pid(&env, 2);

        let first = client.buy_collateral_with_id(&buyer, &id, &purchase, &1_000, &2_000);
        assert_eq!(client.get_auction(&id).status, AuctionStatus::Completed);

        // A terminal auction does not turn a legitimate retry into an error
        let retry = client.buy_collateral_with_id(&buyer, &id, &purchase, &1_000, &2_000);
        assert_eq!(retry, first);
    }

    #[test]
    fn test_purchase_id_conflicting_payloads_rejected() {
        let (env, client, _, _, coll, debt) = setup_env();
        let (id, _) = start_auction_with(&env, &client, &coll, &debt, 1_000, 5_000);
        let (other_id, _) = start_auction_with(&env, &client, &coll, &debt, 1_000, 5_000);
        let buyer = funded_buyer(&env, &debt);
        let other_buyer = funded_buyer(&env, &debt);
        let purchase = pid(&env, 3);

        client.buy_collateral_with_id(&buyer, &id, &purchase, &100, &200);

        let conflict = Err(Ok(AuctionError::PurchaseIdConflict));
        // Different amount
        assert_eq!(
            client.try_buy_collateral_with_id(&buyer, &id, &purchase, &101, &200),
            conflict
        );
        // Different slippage bound
        assert_eq!(
            client.try_buy_collateral_with_id(&buyer, &id, &purchase, &100, &201),
            conflict
        );
        // Different buyer
        assert_eq!(
            client.try_buy_collateral_with_id(&other_buyer, &id, &purchase, &100, &200),
            conflict
        );
        // Different auction
        assert_eq!(
            client.try_buy_collateral_with_id(&buyer, &other_id, &purchase, &100, &200),
            conflict
        );

        // Original fill accounting is preserved
        let auction = client.get_auction(&id);
        assert_eq!(auction.collateral_sold, 100);
        assert_eq!(auction.debt_recovered, 120);
        assert_eq!(client.get_auction(&other_id).collateral_sold, 0);
        assert_eq!(
            client.get_purchase(&purchase).unwrap().requested_collateral,
            100
        );
    }

    #[test]
    fn test_purchase_id_distinct_ids_each_fill() {
        let (env, client, _, _, coll, debt) = setup_env();
        let (id, _) = start_auction_with(&env, &client, &coll, &debt, 1_000, 5_000);
        let buyer = funded_buyer(&env, &debt);

        client.buy_collateral_with_id(&buyer, &id, &pid(&env, 4), &100, &200);
        client.buy_collateral_with_id(&buyer, &id, &pid(&env, 5), &100, &200);

        assert_eq!(client.get_auction(&id).collateral_sold, 200);
    }

    #[test]
    fn test_purchase_id_failed_fill_is_not_recorded() {
        let (env, client, _, _, coll, debt) = setup_env();
        let (id, _) = start_auction_with(&env, &client, &coll, &debt, 1_000, 5_000);
        let buyer = funded_buyer(&env, &debt);
        let purchase = pid(&env, 6);

        // Slippage bound too low: 100 @ 1.2 = 120 > 50
        assert_eq!(
            client.try_buy_collateral_with_id(&buyer, &id, &purchase, &100, &50),
            Err(Ok(AuctionError::InvalidParams))
        );
        assert!(client.get_purchase(&purchase).is_none());

        // The identifier remains usable for a corrected payload
        let fill = client.buy_collateral_with_id(&buyer, &id, &purchase, &100, &200);
        assert_eq!(fill.0, 100);
    }

    // ── #1472: Inventory tracking with a strict upper bound ─────────

    #[test]
    fn test_inventory_partial_fills_by_multiple_buyers() {
        let (env, client, _, _, coll, debt) = setup_env();
        let (id, _) = start_auction_with(&env, &client, &coll, &debt, 1_000, 5_000);
        let a = funded_buyer(&env, &debt);
        let b = funded_buyer(&env, &debt);

        assert_eq!(client.get_remaining_inventory(&id), 1_000);
        client.buy_collateral(&a, &id, &300, &1_000);
        assert_eq!(client.get_remaining_inventory(&id), 700);
        client.buy_collateral(&b, &id, &250, &1_000);
        assert_eq!(client.get_remaining_inventory(&id), 450);

        let auction = client.get_auction(&id);
        assert_eq!(auction.status, AuctionStatus::Active);
        assert_eq!(auction.collateral_sold, 550);

        let coll_client = token::Client::new(&env, &coll);
        assert_eq!(coll_client.balance(&a), 300);
        assert_eq!(coll_client.balance(&b), 250);
        assert_eq!(coll_client.balance(&client.address), 450);
    }

    #[test]
    fn test_inventory_exact_depletion_completes_auction() {
        let (env, client, _, _, coll, debt) = setup_env();
        let (id, _) = start_auction_with(&env, &client, &coll, &debt, 1_000, 5_000);
        let a = funded_buyer(&env, &debt);
        let b = funded_buyer(&env, &debt);
        let c = funded_buyer(&env, &debt);

        client.buy_collateral(&a, &id, &400, &1_000);
        let (received, _) = client.buy_collateral(&b, &id, &600, &1_000);
        assert_eq!(received, 600);

        let auction = client.get_auction(&id);
        assert_eq!(auction.status, AuctionStatus::Completed);
        assert_eq!(auction.collateral_sold, auction.collateral_amount);
        assert_eq!(client.get_remaining_inventory(&id), 0);

        let coll_client = token::Client::new(&env, &coll);
        assert_eq!(coll_client.balance(&client.address), 0);

        // Nothing left: further demand is rejected
        assert_eq!(
            client.try_buy_collateral(&c, &id, &1, &1_000),
            Err(Ok(AuctionError::AuctionNotActive))
        );
    }

    #[test]
    fn test_inventory_excess_demand_is_capped() {
        let (env, client, _, _, coll, debt) = setup_env();
        let (id, _) = start_auction_with(&env, &client, &coll, &debt, 1_000, 5_000);
        let a = funded_buyer(&env, &debt);
        let b = funded_buyer(&env, &debt);

        client.buy_collateral(&a, &id, &700, &2_000);

        // Quote and fill are both bounded by the 300 remaining
        // 300 @ 1.2 = 360 + 18 penalty
        assert_eq!(client.quote_buy(&id, &500), 378);
        let (received, paid) = client.buy_collateral(&b, &id, &500, &2_000);
        assert_eq!(received, 300);
        assert_eq!(paid, 378);

        let auction = client.get_auction(&id);
        assert_eq!(auction.collateral_sold, 1_000);
        assert_eq!(auction.status, AuctionStatus::Completed);

        let coll_client = token::Client::new(&env, &coll);
        assert_eq!(coll_client.balance(&a) + coll_client.balance(&b), 1_000);
        assert_eq!(coll_client.balance(&client.address), 0);
    }

    #[test]
    fn test_inventory_isolated_between_auctions() {
        let (env, client, _, _, coll, debt) = setup_env();
        let (first, _) = start_auction_with(&env, &client, &coll, &debt, 1_000, 5_000);
        let (second, _) = start_auction_with(&env, &client, &coll, &debt, 500, 5_000);
        let buyer = funded_buyer(&env, &debt);

        // Excess demand on the smaller auction cannot reach the other's escrow
        let (received, _) = client.buy_collateral(&buyer, &second, &1_000, &5_000);
        assert_eq!(received, 500);
        assert_eq!(client.get_remaining_inventory(&first), 1_000);

        let coll_client = token::Client::new(&env, &coll);
        assert_eq!(coll_client.balance(&client.address), 1_000);
    }

    // ── #1473: Terminal states and settlement ───────────────────────

    #[test]
    fn test_terminal_states() {
        assert!(!AuctionStatus::Active.is_terminal());
        assert!(AuctionStatus::Completed.is_terminal());
        assert!(AuctionStatus::Cancelled.is_terminal());
    }

    #[test]
    fn test_settle_completed_auction_rejected_without_token_movement() {
        let (env, client, admin, _, coll, debt) = setup_env();
        let (id, vault_owner) = start_auction_with(&env, &client, &coll, &debt, 1_000, 5_000);
        let buyer = funded_buyer(&env, &debt);
        client.buy_collateral(&buyer, &id, &1_000, &5_000);
        assert_eq!(client.get_auction(&id).status, AuctionStatus::Completed);

        env.ledger().set_timestamp(10_000);
        let coll_client = token::Client::new(&env, &coll);
        let debt_client = token::Client::new(&env, &debt);
        let escrow_debt = debt_client.balance(&client.address);

        assert_eq!(
            client.try_settle_auction(&id),
            Err(Ok(AuctionError::AuctionNotActive))
        );
        assert_eq!(
            client.try_cancel_auction(&admin, &id),
            Err(Ok(AuctionError::AuctionNotActive))
        );

        assert_eq!(coll_client.balance(&vault_owner), 0);
        assert_eq!(coll_client.balance(&buyer), 1_000);
        assert_eq!(debt_client.balance(&client.address), escrow_debt);
        assert_eq!(client.get_auction(&id).status, AuctionStatus::Completed);
    }

    #[test]
    fn test_settle_cancelled_auction_rejected_without_token_movement() {
        let (env, client, admin, _, coll, debt) = setup_env();
        let (id, vault_owner) = start_auction_with(&env, &client, &coll, &debt, 1_000, 800);
        client.cancel_auction(&admin, &id);

        env.ledger().set_timestamp(10_000);
        let coll_client = token::Client::new(&env, &coll);

        assert_eq!(
            client.try_settle_auction(&id),
            Err(Ok(AuctionError::AuctionNotActive))
        );
        let buyer = funded_buyer(&env, &debt);
        assert_eq!(
            client.try_buy_collateral(&buyer, &id, &100, &1_000),
            Err(Ok(AuctionError::AuctionNotActive))
        );

        assert_eq!(coll_client.balance(&client.address), 1_000);
        assert_eq!(coll_client.balance(&vault_owner), 0);
        assert_eq!(client.get_auction(&id).status, AuctionStatus::Cancelled);
    }

    #[test]
    fn test_settle_active_auction_after_partial_fill() {
        let (env, client, _, _, coll, debt) = setup_env();
        let (id, vault_owner) = start_auction_with(&env, &client, &coll, &debt, 1_000, 5_000);
        let buyer = funded_buyer(&env, &debt);
        client.buy_collateral(&buyer, &id, &300, &1_000);

        env.ledger().set_timestamp(5_000);
        client.settle_auction(&id);

        let auction = client.get_auction(&id);
        assert_eq!(auction.status, AuctionStatus::Completed);
        assert_eq!(auction.collateral_sold, 300);
        // 300 @ 1.2 = 360 proceeds; 700 collateral unsold
        assert_eq!(client.get_withdrawable(&id), (360, 700));

        // Settlement itself moves no tokens
        let coll_client = token::Client::new(&env, &coll);
        assert_eq!(coll_client.balance(&client.address), 700);
        assert_eq!(coll_client.balance(&vault_owner), 0);

        // Settling twice is rejected
        assert_eq!(
            client.try_settle_auction(&id),
            Err(Ok(AuctionError::AuctionNotActive))
        );
    }

    // ── #1474: Separate seller withdrawals ──────────────────────────

    #[test]
    fn test_withdraw_unauthorized_callers_rejected() {
        let (env, client, admin, _, coll, debt) = setup_env();
        let (id, _) = start_auction_with(&env, &client, &coll, &debt, 1_000, 5_000);
        let buyer = funded_buyer(&env, &debt);
        client.buy_collateral(&buyer, &id, &300, &1_000);
        client.cancel_auction(&admin, &id);

        let stranger = Address::generate(&env);
        for caller in [stranger, buyer, admin] {
            assert_eq!(
                client.try_withdraw_proceeds(&caller, &id),
                Err(Ok(AuctionError::Unauthorized))
            );
            assert_eq!(
                client.try_withdraw_unsold(&caller, &id),
                Err(Ok(AuctionError::Unauthorized))
            );
        }
        assert_eq!(client.get_withdrawable(&id), (360, 700));
    }

    #[test]
    fn test_withdraw_requires_seller_auth() {
        let (env, client, _, _, coll, debt) = setup_env();
        let (id, vault_owner) = start_auction_with(&env, &client, &coll, &debt, 1_000, 5_000);
        let buyer = funded_buyer(&env, &debt);
        client.buy_collateral(&buyer, &id, &300, &1_000);

        client.withdraw_proceeds(&vault_owner, &id);
        let auths = env.auths();
        assert_eq!(auths.len(), 1);
        assert_eq!(auths[0].0, vault_owner);
    }

    #[test]
    fn test_withdraw_early_rejected() {
        let (env, client, _, _, coll, debt) = setup_env();
        let (id, vault_owner) = start_auction_with(&env, &client, &coll, &debt, 1_000, 5_000);

        // No fills yet: no proceeds
        assert_eq!(
            client.try_withdraw_proceeds(&vault_owner, &id),
            Err(Ok(AuctionError::NothingToWithdraw))
        );

        let buyer = funded_buyer(&env, &debt);
        client.buy_collateral(&buyer, &id, &300, &1_000);

        // Inventory is still for sale while active, even after expiry
        assert_eq!(
            client.try_withdraw_unsold(&vault_owner, &id),
            Err(Ok(AuctionError::AuctionStillActive))
        );
        env.ledger().set_timestamp(10_000);
        assert_eq!(
            client.try_withdraw_unsold(&vault_owner, &id),
            Err(Ok(AuctionError::AuctionStillActive))
        );
        assert_eq!(client.get_withdrawable(&id), (360, 0));

        // Proceeds are final once filled and can be withdrawn mid-auction
        assert_eq!(client.withdraw_proceeds(&vault_owner, &id), 360);
    }

    #[test]
    fn test_withdraw_repeated_rejected() {
        let (env, client, _, _, coll, debt) = setup_env();
        let (id, vault_owner) = start_auction_with(&env, &client, &coll, &debt, 1_000, 5_000);
        let buyer = funded_buyer(&env, &debt);
        let debt_client = token::Client::new(&env, &debt);
        let coll_client = token::Client::new(&env, &coll);

        client.buy_collateral(&buyer, &id, &300, &1_000);
        assert_eq!(client.withdraw_proceeds(&vault_owner, &id), 360);
        assert_eq!(
            client.try_withdraw_proceeds(&vault_owner, &id),
            Err(Ok(AuctionError::NothingToWithdraw))
        );

        // New fills accrue only the new delta
        client.buy_collateral(&buyer, &id, &100, &1_000);
        assert_eq!(client.withdraw_proceeds(&vault_owner, &id), 120);
        assert_eq!(debt_client.balance(&vault_owner), 480);

        env.ledger().set_timestamp(10_000);
        client.settle_auction(&id);
        assert_eq!(client.withdraw_unsold(&vault_owner, &id), 600);
        assert_eq!(
            client.try_withdraw_unsold(&vault_owner, &id),
            Err(Ok(AuctionError::NothingToWithdraw))
        );

        assert_eq!(coll_client.balance(&vault_owner), 600);
        assert_eq!(coll_client.balance(&client.address), 0);
        assert_eq!(debt_client.balance(&client.address), 0);
        assert_eq!(client.get_withdrawable(&id), (0, 0));
    }

    #[test]
    fn test_withdraw_unsold_after_full_sale_has_nothing() {
        let (env, client, _, _, coll, debt) = setup_env();
        let (id, vault_owner) = start_auction_with(&env, &client, &coll, &debt, 1_000, 5_000);
        let buyer = funded_buyer(&env, &debt);
        client.buy_collateral(&buyer, &id, &1_000, &5_000);

        assert_eq!(
            client.try_withdraw_unsold(&vault_owner, &id),
            Err(Ok(AuctionError::NothingToWithdraw))
        );
        assert_eq!(client.withdraw_proceeds(&vault_owner, &id), 1_200);
    }
}
