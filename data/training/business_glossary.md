# Business Glossary

These notes are schema-derived guidance for SQL generation. Use direct table and view names without `public.` prefixes. Prefer read-only analytical SQL.

Treat similarly named tables with care. Similar names do not imply a valid join path, and an empty result does not prove the table is wrong.

---

General query rules:

- Use unqualified table and view names such as `player`, `game`, `player_skills`, and `leaderboard_past`.
- `player.id` is the canonical player key across most player-facing tables.
- If the question needs `username`, `email`, provider, or geography, join to `player` unless the target view already exposes that field.
- Prefer the table or view that already contains the requested metric instead of inventing an extra join to a similarly named object.
- Do not treat an empty result as proof that a feature or table has no data. First check whether the filter, date range, or join path was too narrow.
- Prefer curated views when they already expose parsed JSON or pre-aggregated metrics.
- Prefer read-only SQL. Do not generate `INSERT`, `UPDATE`, `DELETE`, `TRUNCATE`, or DDL for normal question answering.

---

Player and account domain:

- `player` is the core account and profile table.
- `player` stores identity, provider, availability, gameplay status, geography, OS, version, and account timestamps.
- `player` includes both `country_id` and `continent_id`.
- Join `player.provider_id -> login_provider.id` for provider labels.
- Join `player.availability_id -> availability.id` for availability labels.
- Join `player.gameplay_status_id -> gameplay_status.id` for gameplay labels.
- Join `player.country_id -> country.id` for country names.
- Join `player.continent_id -> continent.id` for continent names.
- `player_login` is the login event table and should be used for login-event analysis.
- `player.last_online` is a last-known activity timestamp, not a login event stream.
- `linked_account` stores provider-linked account payloads for players.

---

Preference and settings domain:

- `user_preference` stores gameplay preferences such as `game_speed`.
- `user_settings` stores UX toggles such as sound, vibration, timer style, push notification state, background theme, and card theme.
- `view_user_pref` is a curated preference view and only exposes `player_id`, `background_theme`, and `card_theme`.

---

Gameplay domain:

- `game` is the main gameplay fact table.
- Join `game.player_id -> player.id`, `game.game_mode_id -> game_mode.id`, and `game.game_scope_id -> game_scope.id`.
- `game.completed` and `game.canceled` are separate flags and should not be assumed to be perfect inverses.
- `game.players`, `game.all_ranks`, `game.scores`, and `game.all_skills` are JSON payloads that describe participants and outcomes.
- `game_data` stores match payload details keyed by `match_id`.
- `top_skilled_player_games` is a curated gameplay-oriented object that already contains `username`, `player_id`, `skill`, `highest_bid`, `started_at`, mode and scope ids, and gameplay payload fields.
- Use `top_skilled_player_games` when the question is explicitly about sampled top-skilled game rows, last sampled games, or payloads already present there.
- `view_day3_games_played` and `view_day7_games_played` are curated onboarding-engagement views with quoted display-style column names.

---

Statistics, skills, achievements, and leaderboard domain:

- `statistics` is an aggregate performance table keyed by player, mode, scope, and `time_range_id`.
- `player_skills` is the preferred curated view for current human-versus-human skill and matchmaking latency.
- `player_skills` exposes `mu`, `sigma`, `total_games`, `total_completed`, and wait-time percentiles such as `wait_time_p25`, `wait_time_p50`, `wait_time_p75`, `wait_time_p99`, and `max_wait_time`.
- `achievements` stores rank and percentile data by metric, geography, mode, scope, and player.
- `leaderboard_past` is a leaderboard snapshot table with numeric measures such as `skill`, `highest_bid`, `win_ratio`, and rank and percentile fields, plus `metric_id`, `game_mode_id`, `game_scope_id`, `country_id`, `continent_id`, and `player_id`.
- `leaderboard_metadata` tracks leaderboard refresh state and timestamps.
- `leaderboard1`, `leaderboard2`, and `leaderboard_past` are similarly named and should not be swapped casually.
- If the question asks for current skill ranking, prefer `player_skills`.
- If the question asks for stored leaderboard snapshot values, world rank, country rank, continent rank, or other rank columns, use `leaderboard_past`.
- For usernames on leaderboard rows, join `leaderboard_past.player_id = player.id`.
- Do not join `leaderboard_past` to `top_skilled_player_games` just to find `username`. `top_skilled_player_games` is a different object with its own purpose.

---

Matchmaking domain:

- `match_making`, `tickets`, `ticket_guest`, `stake_ticket`, and `ticket_activity` are matchmaking objects, not customer-support tickets.
- `ticket_activity` is useful for wait-time, disconnect, reconnect, and matchmaking outcome analysis.
- `ticket_activity.uid` is text. When joining it to `player`, cast with `player.id::text`.
- `view_player_with_mm_issue`, `view_users_each_issue`, and `view_each_players_issue_count` are matchmaking issue-analysis views.

---

Social and referral domain:

- `friend_requests` stores sender and receiver pairs plus status.
- `friend` stores connected relationships and `connected_date`.
- `referral` links `referred_by` and `referred_to` back to `player.id`.
- `referral_reward` is reward configuration, while `referral` is the activity fact table.

---

Currency, rewards, and progression domain:

- `player_vault` stores a current gem-balance row per player and includes `gems`, `unclaimed_gems`, and `unprocessed_playsuper_gems`.
- `wallet` stores a current balance row per player and includes both `coin` and `gem`.
- `gem_transaction` is the gem movement history table.
- `daily_reward` stores per-player reward streak state.
- `daily_rewards_transaction` stores reward claim history and joins to `rewards`.
- `rewards` stores reward configuration such as `reward_type`, `reward_version`, `coin_value`, and `gem_value`.

---

Store, offers, purchases, and campaigns domain:

- `offer`, `store_item`, `asset`, `gem_packs`, `iap_products`, and `tags` are store and catalog configuration objects.
- `store_item` links to both `asset` and `offer`.
- `campaigns` stores live-ops campaign definitions with date windows, targeting, and version constraints.
- `campaign_assets` maps campaigns to product assets.
- `app_store_purchases` is raw purchase fact data.
- `purchase_status` stores raw purchase payload JSON.
- `product_purchases` is the parsed purchase view and is usually easier to query than raw payload JSON.
- `view_yesterday_iap` is a curated prior-day IAP view and already exposes player and provider fields.
- `voided_purchases` and `purchase_transfer` are separate purchase lifecycle tables.

---

PlaySuper domain:

- `playsuper_transaction` and `playsuper_all_transaction` store PlaySuper coin movement and delta data.
- `playsuper_brands` stores PlaySuper brand metadata, listing state, and priority.
- `player_vault.unprocessed_playsuper_gems` connects PlaySuper activity to player-balance context.

---

Feedback, support, app telemetry, and operations domain:

- `app_rating` stores player star ratings and client context.
- `bug_report` is the main free-text player issue table.
- `support_message` stores support and purchase issue messages and is separate from matchmaking ticket tables.
- `api_profile` is application API performance and status telemetry.
- `app_install` is install telemetry by platform and version code.

---

Events, surveys, themes, and notifications domain:

- `events`, `events_login`, `aniversary_login`, and `user_events` capture event participation and claim state.
- The table name is spelled `aniversary_login` in the schema and should be referenced exactly that way in SQL.
- `surveys` stores survey definitions, while `survey_responses` stores response payloads and entry-point context.
- `themes` defines theme availability windows and remote-config dependency paths.
- `holiday`, `notification`, and `version` support holiday notifications by country, OS, version, and trigger time.

---

Views to prefer for training examples:

- `player_skills` for current human-versus-human skill plus wait-time percentiles.
- `product_purchases` for parsed purchase JSON fields.
- `view_yesterday_iap` for prior-day IAP slices.
- `view_user_pref` for theme-preference questions.
- `view_day3_games_played` and `view_day7_games_played` for early-engagement questions.

---

Views and patterns to use with caution:

- Diagnostic views with embedded fixed dates, such as `view_top_20_p_id`, `view_top_20_90_days`, `view_top_20_last_90_days_median_wt_twt`, and `view_5_20_top_g_status`.
- Database diagnostic objects such as `pg_stat_statements` and `pg_stat_statements_info`, which are not gameplay or business entities.
- Sparse or empty views. A zero-row result should not be narrated as "the table has no data" unless the schema and filters were checked carefully first.
