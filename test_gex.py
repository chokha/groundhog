from orats import fetch_chain
from gex import compute_gex_map

# Fetch QQQ chain for Nov 14 2025
df = fetch_chain("QQQ", "2025-11-14")

# QQQ spot on Nov 14 2025 was approximately 493.50
spot = 493.50

result = compute_gex_map(df, spot, label="qqq")

print(f"Rows used in GEX calc: {len(result['by_strike'])}")
print(f"Total net GEX: {result['by_strike']['gex'].sum():,.0f}")
print(f"Strikes above spot: {len(result['by_strike'][result['by_strike']['strike'] > spot])}")
print(f"Strikes below spot: {len(result['by_strike'][result['by_strike']['strike'] < spot])}")
print()
print(f"Spot:        {spot}")
print(f"Gamma Flip:  {result['gamma_flip']}")
print(f"Call Wall:   {result['call_wall']}")
print(f"Put Wall:    {result['put_wall']}")
print(f"Air Up:      {result['air_up']} pts")
print(f"Air Down:    {result['air_dn']} pts")
print(f"Regime:      {result['regime']}")
print(f"Support below spot: {result.get('support_below')}")
print(f"Support above spot: {result.get('support_above')}")
print(f"All nodes above spot: {result.get('all_nodes_above_spot')}")
print(f"Wall Magnetism:      {result['wall_magnetism']}")
print(f"Walls above spot:    {result['walls_above_count']}")
print(f"Walls below spot:    {result['walls_below_count']}")
print(f"Wall imbalance:      {result['wall_imbalance']}")
print(f"Nearest wall above:  {result['nearest_wall_above']}")
print(f"Nearest wall below:  {result['nearest_wall_below']}")
print(f"BIAS:                {result['bias']}")
print(f"\nTop GEX nodes:")
top = result['by_strike'].nlargest(5, 'gex')[['strike','gex']]
bot = result['by_strike'].nsmallest(5, 'gex')[['strike','gex']]
import pandas as pd
print(pd.concat([top, bot]).sort_values('strike').to_string(index=False))
