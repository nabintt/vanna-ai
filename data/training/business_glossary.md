# Business Glossary for prod_clone

These notes were inferred on 2026-04-27 by inspecting the `public` schema, enum values, foreign keys, indexes, and view definitions in `prod_clone`.

Treat this as schema-derived training context, not product-owner-validated business truth. Validate naming, metric definitions, and any inferred semantics before relying on them for production analytics.

---

General query rules:

- Prefer fully qualified table and view names in the `public` schema.
- `player.id` is the canonical player key across most player-facing tables.
- Prefer read-only analytical queries.
- Prefer curated views when they already expose parsed JSON or pre-aggregated metrics.
- Avoid one-off diagnostic views with embedded historical dates unless the question explicitly asks for those objects.

---

Repository-wide schema context:

- The `public` schema contains 99 base tables and 26 views in this clone.
- No table or column comments were present, so business meaning here is inferred from names, keys, enum labels, and view SQL.
- Some configuration or staging tables appear sparse in this clone. Absence of rows should not be interpreted as absence of the feature without validation.

---

Player and account domain:

- `public.player` is the core player/account entity.
- `player` stores profile fields, provider, geography, availability, gameplay status, OS, version, and account timestamps.
- Join `player.provider_id -> login_provider.id` for provider labels: `GUEST`, `GOOGLE`, `FACEBOOK`, `APPLE`, `TWITTER`.
- Join `player.availability_id -> availability.id` for availability labels: `OFFLINE`, `ONLINE`, `DND`, `INVISIBLE`, `AWAY`.
- Join `player.gameplay_status_id -> gameplay_status.id` for gameplay labels currently stored in the lookup table: `IDLE`, `IN_GAME`, `IN_LOBBY`.
- Join `player.country_id -> country.id` and `country.continent_id -> continent.id` for geography.
- `player.identifier` is a UUID-style identifier, while `player.id` is the main integer PK used across the schema.
- `player_login` is the login event table and should be used for login-event analysis.
- `player.last_online` is a last-known timestamp, not a login event stream.
- `linked_account` stores provider-linked account payloads for players.

---

Preference and settings domain:

- `user_preference` stores gameplay toggles like autoplay, reduced ads preference, and `game_speed`.
- The observed `game_speed` enum values are `SLOWER`, `NORMAL`, and `FAST`.
- `user_settings` stores UI and UX settings such as sound, vibration, push notifications, timer style, background theme, and card theme.
- `view_user_pref` exposes players with non-null background or card theme values.

---

Gameplay domain:

- `public.game` is the main gameplay fact table.
- Join `game.player_id -> player.id`, `game.game_mode_id -> game_mode.id`, `game.game_scope_id -> game_scope.id`, and `game.game_data_id -> game_data.id`.
- `game_mode` values include `STANDARD`, `QUICK`, `EIGHT_BID_CALL`, `EIGHT_BID_BREAK`, `TRANING`, and `TUTORIAL`.
- `game_scope` values include `VS_BOTS`, `VS_HUMANS`, `PRIVATE`, `LAN`, and `CHALLENGE`.
- `game.completed` and `game.canceled` are separate booleans. Do not assume they are perfect inverses without validating application logic.
- `game.players`, `game.all_ranks`, `game.scores`, and `game.all_skills` are JSON payloads that describe match participants and outcomes.
- `game_data` contains array-heavy match payload details keyed by `match_id`.
- `view_day3_games_played` and `view_day7_games_played` are reusable onboarding-engagement views, but their column names are quoted display-style labels with spaces.

---

Statistics, skills, achievements, and leaderboard domain:

- `statistics` is an aggregate player-performance table keyed by player, mode, scope, and `time_range_id`.
- `player_skills` is a curated analytics view for current human-versus-human skill and matchmaking latency.
- `player_skills` filters `statistics.game_scope_id = 2` (`VS_HUMANS`) and `statistics.time_range_id = 0`, then adds wait-time percentiles from `ticket_activity`.
- `achievements` stores rank and percentile data by metric, geography, mode, scope, and player.
- `metrics` values include `SKILL`, `HIGHEST_SCORE`, `AVERAGE_SCORE`, `LOWEST_SCORE_TO_WIN`, `WIN_RATIO`, `HIGHEST_BID`, `MAX_WINNING_STREAK`, `MAX_LOSING_STREAK`, `COMPLETION_RATIO`, and `RESPONSE_TIME`.
- `leaderboard1`, `leaderboard2`, and `leaderboard_past` have an achievements-like shape.
- `leaderboard_metadata` tracks refresh timestamps and leaderboard processing state.
- The semantic difference between `leaderboard1` and `leaderboard2` is not documented in schema names alone, so prefer `statistics`, `achievements`, or `player_skills` unless a question explicitly asks for those tables.

---

Matchmaking domain:

- `match_making`, `tickets`, `ticket_guest`, `stake_ticket`, and `ticket_activity` are matchmaking objects, not customer support tables.
- `ticket_activity.status` values are `MERGED`, `CANCELLED`, and `NOTFOUND`.
- `ticket_activity.uid` is text. When joining it to `player`, cast with `player.id::text`.
- `tickets` and `ticket_guest` represent open or historical ticket rows with gameplay-fit metrics like skill, response time, and completion ratio.
- `view_player_with_mm_issue`, `view_users_each_issue`, and `view_each_players_issue_count` are reusable issue-analysis views.
- `view_top_20_p_id`, `view_top_20_90_days`, `view_top_20_last_90_days_median_wt_twt`, and `view_5_20_top_g_status` contain embedded historical date logic and should be treated as one-off diagnostic views.

---

Social and referral domain:

- `friend_requests` stores sender and receiver pairs plus `status_id -> friend_request_status.id`.
- `friend_request_status` values are `PENDING`, `ACCEPTED`, `REJECTED`, `BLOCKED`, `UNFRIENDED`, and `CANCELED`.
- `friend` represents connected relationships, with `connected_date` as the relationship timestamp.
- `referral` links `referred_by` and `referred_to` back to `player.id`.
- Observed live `referral.status` values in this clone are `PENDING` and `CLAIMED`.
- `referral_reward` stores reward-value configuration, but `referral.reward_value` is also present on the fact table.

---

Currency, rewards, and progression domain:

- `player_vault` stores one current gem-balance row per player. There is a unique index on `player_vault.player_id`.
- `player_vault` includes `gems`, `unclaimed_gems`, and `unprocessed_playsuper_gems`.
- `wallet` also stores one current row per player and includes both `gem` and `coin`. There is a unique index on `wallet.player_id`.
- `gem_transaction` is the gem movement history table.
- `daily_reward` stores per-player reward streak state and is unique on `player_id`.
- `daily_rewards_transaction` stores reward claim history and joins to `rewards`.
- `rewards` stores reward configuration fields such as `reward_type`, `reward_version`, `coin_value`, `gem_value`, and reward multipliers.

---

Store, offers, purchases, and campaigns domain:

- `offer`, `store_item`, `asset`, `gem_packs`, `iap_products`, and `tags` describe store and catalog configuration.
- `store_item` links to both `asset` and `offer`.
- `campaigns` stores live-ops campaign definitions with start and end windows, target audience, and version constraints.
- `campaign_assets` maps `campaigns.id` to `asset.product_id`.
- `campaigns.target_audience` values are `GENERAL` and `PLAY_SUPER`.
- `store_campaign` is a separate store-promo asset table with URI windows.
- `app_store_purchases` is a purchase fact table with observed statuses `CANCELED`, `PURCHASED`, and `PENDING`.
- Observed `app_store_purchases.purchase_source` values are `INAPP` and `PLAYGAP`.
- `purchase_status` stores raw purchase payload JSON.
- `product_purchases` is the parsed view over `purchase_status` and exposes fields like `purchase_state`, `purchase_type`, `order_id`, and `region_code`.
- `view_yesterday_iap` is already filtered to yesterday and maps provider and purchase-state codes to readable labels.
- `voided_purchases` stores voided purchase payloads.
- `purchase_transfer` tracks product ownership transfers between players.

---

PlaySuper domain:

- `playsuper_transaction` and `playsuper_all_transaction` store PlaySuper coin movements, balances, reasons, and event types.
- `playsuper_brands` stores brand metadata, listing state, image links, and priority.
- `player_vault.unprocessed_playsuper_gems` suggests PlaySuper activity can affect player gem balances.

---

Feedback, support, app telemetry, and operations domain:

- `app_rating` stores player star ratings plus provider, OS, version, device, and redirect fields.
- `bug_report` is the main free-text player issue table with `subject` and `body`.
- `view_ads`, `view_ads_filter_1`, `view_ads_filter_2`, and `view_ads_filter_3` are text-filtering support views for ad-related issues.
- `support_message` stores purchase and support issues and is separate from matchmaking ticket tables.
- `api_profile` is operational telemetry for application APIs and endpoints.
- `api_profile.api_name` and `api_profile.api_status` are enums, so this table is better for app-operation analysis than for gameplay facts.
- `app_install` is install telemetry by platform and version.

---

Events, surveys, themes, and notifications domain:

- `events` stores event names, active dates, assets, and trigger dates.
- `events_login`, `aniversary_login`, and `user_events` capture per-player event participation and claim state.
- The table name is spelled `aniversary_login` in the schema and should be referenced exactly that way in SQL.
- `surveys` stores survey definitions, while `survey_responses` stores response payloads, entry points, and attached user context.
- `themes` defines theme availability windows and remote-config dependency paths.
- `holiday`, `notification`, and `version` support holiday notifications by country, OS, version, and trigger time.

---

Views to prefer for training examples:

- `public.player_skills` for current human-versus-human skill plus wait-time percentiles.
- `public.product_purchases` for parsed purchase JSON fields.
- `public.view_yesterday_iap` for prior-day IAP slices.
- `public.view_user_pref` for theme-preference questions.
- `public.view_day3_games_played` and `public.view_day7_games_played` for early-engagement questions.

---

Views to use with caution:

- `public.view_top_20_p_id`, `public.view_top_20_90_days`, `public.view_top_20_last_90_days_median_wt_twt`, and `public.view_5_20_top_g_status` contain hardcoded 2025 date logic.
- `public.pg_stat_statements` and `public.pg_stat_statements_info` are database diagnostics, not product/business objects.
