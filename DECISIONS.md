# DECISIONS.md

1. **Indexes.** I use a unique `employees.emp_code` index for identity, and department/join-date indexes for employee and headcount queries. Attendance uses a unique `(emp_code, date)` index for race-safe writes, `(date, emp_code)` for list ordering, and date-led indexes for status and late-month filters. I considered one index per analytics combination, but rejected the extra write/storage cost without a distinct query shape.
2. **Punch-in race.** The unique attendance index arbitrates concurrent inserts: one succeeds; the other gets `DuplicateKeyError`, returned as 409.
3. **Ties.** `$rank` gives equal totals the same rank and skips the next. Filtering by rank returns every employee tied at the cutoff; results sort by total minutes and `emp_code`.
4. **Headcount.** Department summary starts with employees joined by month end, groups them before looking up attendance, and preserves groups without logs. They count toward headcount with zero attendance totals.
5. **100x data.** I would benchmark query load and consider maintained daily rollups for expensive department and leaderboard analytics, retaining source logs for auditability and corrections.

