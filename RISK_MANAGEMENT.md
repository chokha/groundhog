# Risk Management Rules — Non-Negotiable

**Account:** Topstep $50k | **Size:** 15 MNQ | **Daily Loss Limit:** $1,000

These rules are absolute. No exceptions. No "just this once." Print this and tape it to your monitor.

---

## Rule 1: Bracket Orders Only

Every trade enters with an OCO bracket. Target and stop set BEFORE entry. You do not touch them after.

- **ORB trades:** Target = T1 from brief, Stop = below/above ORB range
- **Scalp trades:** +10 pts / -10 pts, hard, no adjustments

If your platform doesn't support brackets, you don't trade.

---

## Rule 2: Morning Decision — One Mode Per Day

Run the brief at 09:25. Run again at 09:34. Pick ONE mode for the day.

```
09:34 brief says RUNNER?
  → ORB mode. One trade. Done after T1 or stop.
  → Win = done for the day.
  → Stop = done for the day. NO switching to scalps.

09:34 brief says SCALP / PIN_FADE?
  → Scalp mode. Max 3 trades. 15 MNQ, 10/10.
  → Hit $900 = done.
  → 3 losses = done.

09:34 brief says TRAP?
  → No trading. Close the platform.
```

**You NEVER switch modes mid-day.**

---

## Rule 3: Daily Loss Circuit Breaker

| Event | Action |
|---|---|
| +$900 or more | **DONE.** Close platform. |
| +$1,200 or more (ORB day) | **DONE.** Close platform. |
| 2 consecutive losses | **DONE.** Close platform. |
| 3 total losses | **DONE.** Close platform. |
| -$900 | **DONE.** Close platform. Hard limit. |
| Hit Topstep daily limit (-$1,000) | Account frozen anyway. |

---

## Rule 4: No Trading After 14:00 ET

Your edge is the morning. ORB window is 09:30-10:00. Scalp window is 10:00-14:00. After 14:00, close the platform. The afternoon is where good mornings go to die.

---

## Rule 5: No Revenge Trading

**If you get stopped out, the next trade must be BETTER than the last one, not faster.**

- Lost on ORB → done for the day (Rule 2)
- Lost on scalp → wait minimum 15 minutes before next entry
- Lost 2 in a row → done for the day (Rule 3)
- "I need to make it back" → that thought means you're done for the day

---

## Rule 6: Size Is Fixed

**15 MNQ. Every trade. No exceptions.**

- Do not size up after a win ("I'm playing with house money")
- Do not size up after a loss ("I need to make it back faster")
- Do not size down ("Just a small one to test")

15 MNQ. Period. Until you have $2,000+ profit buffer above starting balance.

---

## Rule 7: Weekly Loss Limit

| Event | Action |
|---|---|
| Down $1,500 for the week | **No trading Thursday/Friday.** Review journal. |
| Down $1,800 for the week | **No trading rest of week.** |
| Account balance below $48,500 | **Stop trading. Review everything.** |

Topstep max loss is $2,000. You never want to get close. The $48,500 line gives you $500 buffer before the account is gone.

---

## Rule 8: TRAP Mode = No Trading

When the brief says TRAP:
- Do not scalp
- Do not "just take one quick trade"
- Do not watch the screen hoping for a setup
- Close the platform and do something else

TRAP mode exists because the structure creates violent chop that stops you out in both directions. Your 10-point stop is inside a single candle. There is no edge.

---

## Rule 9: The 50-Minute Rule

No single trade should last longer than 10 minutes for scalps. If you're in a scalp and it hasn't hit +10 or -10 in 10 minutes, **close it at market.**

For ORB trades, the window is longer (up to 30 minutes). But if the ORB trade hasn't moved toward T1 within 15 minutes of entry, tighten the stop to breakeven.

Your blowup was a 50-minute hold on a 10-point scalp. That's not a scalp anymore. That's hope.

---

## Rule 10: Journal Every Day You Trade

```bash
python dashboard.py --journal --date YYYY-MM-DD
```

Then add your actual trades:
```bash
python journal.py --date YYYY-MM-DD \
  --macro "what happened in the market" \
  --lesson "what you learned about your execution"
```

No journal = no trading tomorrow. The journal is what feeds the learning system. Without it, you're just gambling with a fancy dashboard.

---

## The Scaling Plan

Only increase size after PROVING the rules work:

| Milestone | Size | Per Win | Trades to $1k |
|---|---|---|---|
| Start (now) | 15 MNQ | $300 | 3-4 |
| +$2,000 profit buffer | 20 MNQ | $400 | 3 |
| +$4,000 profit buffer | 25 MNQ | $500 | 2 |
| Move to $100k account | 25 MNQ | $500 | 2 (with $2k daily limit) |

**You earn bigger size. You don't decide it after a bad day.**

---

## Daily Checklist

Before market open:
- [ ] Run dashboard brief at 09:25
- [ ] Identify mode: RUNNER / SCALP / TRAP
- [ ] If TRAP: close platform, done
- [ ] Set bracket orders ready (10/10 for scalps, T1/stop for ORB)

During session:
- [ ] Maximum 1 ORB trade OR 3 scalp trades
- [ ] Bracket order on EVERY entry
- [ ] Never switch modes mid-day
- [ ] Check P&L only at end of session, not during

After close:
- [ ] Journal the day
- [ ] Review: did I follow all 10 rules?
- [ ] If any rule was broken: why? Write it in the lesson.

---

## The Only Number That Matters

**$900/day x 20 trading days = $18,000/month**

You don't need to hit $1,000 every day. You don't need to trade every day. You need to survive long enough for the edge to compound.

3 good trades at $300 each. That's the whole game. Everything else is noise.
