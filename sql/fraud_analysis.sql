
-- name: 01_overview
-- SELECT basics + aggregates
SELECT COUNT(*)                         AS transactions,
       SUM(isFraud)                     AS frauds,
       ROUND(100.0 * AVG(isFraud), 3)   AS fraud_rate_pct,
       ROUND(SUM(TransactionAmt), 0)    AS total_amount,
       ROUND(SUM(CASE WHEN isFraud = 1 THEN TransactionAmt END), 0) AS fraud_amount
FROM txn;

-- name: 02_by_product
-- GROUP BY + HAVING (ignore tiny groups)
SELECT ProductCD,
       COUNT(*)                       AS n,
       ROUND(100.0 * AVG(isFraud), 2) AS fraud_rate_pct,
       ROUND(AVG(TransactionAmt), 1)  AS avg_amount
FROM txn
GROUP BY ProductCD
HAVING COUNT(*) > 500
ORDER BY fraud_rate_pct DESC;

-- name: 03_by_hour
-- When do frauds happen? (hour of day, derived column)
SELECT hour,
       COUNT(*)                       AS n,
       ROUND(100.0 * AVG(isFraud), 2) AS fraud_rate_pct
FROM txn
GROUP BY hour
ORDER BY hour;

-- name: 04_identity_effect
-- NULL handling: has_identity = 0 means no identity row exists (LEFT JOIN gave NULLs)
SELECT has_identity,
       COUNT(*)                       AS n,
       ROUND(100.0 * AVG(isFraud), 2) AS fraud_rate_pct
FROM txn
GROUP BY has_identity;

-- name: 05_email_lift
-- CTE + subquery: which e-mail domains are riskier than average? (lift = domain rate / overall rate)
WITH overall AS (
    SELECT AVG(isFraud) AS rate FROM txn
),
by_domain AS (
    SELECT COALESCE(P_emaildomain, 'missing') AS domain,
           COUNT(*)                           AS n,
           AVG(isFraud)                       AS rate
    FROM txn
    GROUP BY COALESCE(P_emaildomain, 'missing')
    HAVING COUNT(*) >= 300
)
SELECT d.domain, d.n,
       ROUND(100.0 * d.rate, 2)        AS fraud_rate_pct,
       ROUND(d.rate / o.rate, 2)       AS lift
FROM by_domain d CROSS JOIN overall o
ORDER BY lift DESC
LIMIT 10 ;

-- name: 06_daily_trend
-- Window function: 7-day moving average of the daily fraud rate
WITH daily AS (
    SELECT day, COUNT(*) AS n, AVG(isFraud) AS rate
    FROM txn
    GROUP BY day
)
SELECT day, n,
       ROUND(100.0 * rate, 2) AS fraud_rate_pct,
       ROUND(100.0 * AVG(rate) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS moving_avg_7d_pct
FROM daily
ORDER BY day;

-- name: 07_top_domains_per_product
-- Window function: RANK inside each group, then keep the top 3 (interview classic)
WITH stats AS (
    SELECT ProductCD,
           COALESCE(P_emaildomain, 'missing') AS domain,
           COUNT(*)      AS n,
           AVG(isFraud)  AS rate
    FROM txn
    GROUP BY ProductCD, COALESCE(P_emaildomain, 'missing')
    HAVING COUNT(*) >= 100
),
ranked AS (
    SELECT *, RANK() OVER (PARTITION BY ProductCD ORDER BY rate DESC) AS rnk
    FROM stats
)
SELECT ProductCD, domain, n, ROUND(100.0 * rate, 2) AS fraud_rate_pct, rnk
FROM ranked
WHERE rnk <= 3
ORDER BY ProductCD, rnk;

-- name: 08_amount_deciles
-- NTILE: split transactions into 10 equal-size amount buckets
WITH bucketed AS (
    SELECT isFraud, TransactionAmt, NTILE(10) OVER (ORDER BY TransactionAmt) AS decile
    FROM txn
)
SELECT decile,
       ROUND(MIN(TransactionAmt), 2)  AS min_amt,
       ROUND(MAX(TransactionAmt), 2)  AS max_amt,
       ROUND(100.0 * AVG(isFraud), 2) AS fraud_rate_pct
FROM bucketed
GROUP BY decile
ORDER BY decile;

-- name: 09_card_type_mix
-- Two-column GROUP BY: fraud rate for each card network x debit/credit combination
SELECT card4, card6,
       COUNT(*)                       AS n,
       ROUND(100.0 * AVG(isFraud), 2) AS fraud_rate_pct
FROM txn
WHERE card4 IS NOT NULL AND card6 IS NOT NULL
GROUP BY card4, card6
HAVING COUNT(*) > 200
ORDER BY fraud_rate_pct DESC;
