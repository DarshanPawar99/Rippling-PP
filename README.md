Rippling Cafeteria Production Optimiser
Predicts ideal per-pax food portions with a TensorFlow embedding model and turns those predictions into three actionable production plans.

Key Features
Weekday interaction – learns the typical Monday-to-Sunday demand rhythm.
Item interaction – models cross-dish effects using a 10-item context window (e.g. biryani ↑ ⇒ dal ↓).
Month excluded – avoids calendar noise due to data constrents; keeps the model focused on weekday and menu mix.
Sub-category fallback – unseen dishes map to a proxy from the same sub-category instead of failing.
Holiday shortcut – table-driven MG reduction you can apply without rerunning the model.

Model in a Nutshell
Inputs : target_item, 10×context, sub_cat, cat, weekday, day_type, holiday_type
Embed  : 8-4-2 dims, context averaged
Dense  : 64 → 32 + 0.3 dropout
Output : ideal_per_pax_qty   (MSE loss, Adam optimiser)

Post-Prediction Calculations
per_pax_r   = floor(pred / 0.005) * 0.005
vendor_PP   = ceil(per_pax_r + 0.01, 2 dp)
total_qty   = per_pax_r × client_MG
item_MG     = total_qty / vendor_PP

Final Vendor MG
Non-veg category – one user-chosen category
Star items – Flavoured Rice & Veg Gravy
Remaining items – everything else
final_vendor_MG = mean(MG_nonveg, MG_star_items, MG_remaining)

Aggressive Plan & Slab-Bump Logic
The aggressive vendor plan adds a controlled cushion to star items and the chosen non-veg category.
Spike definition – first raise the ordered weight by 10 %.
Spike size = spiked weight – original weight.
Extra bump is a fraction of that spike size:

Spike size (kg)	Extra added
≤ 2 kg	100 % of spike size
≤ 4 kg	35 % of spike size
≤ 6 kg	25 % of spike size
≤ 8 kg	15 % of spike size
> 8 kg	10 % of spike size

The final aggressive ordered weight is
ordered_qty_aggr = round(original + extra, 1).
Vendor PP is then re-calculated on a tighter 0.005 grid, and MG is re-balanced and rounded to the nearest 5 g.

Output Tables
Client Plan – category · item · client PP · total kg
Vendor Plan (regular) – vendor PP · ordered kg · vendor MG
Vendor Plan (aggressive) – slab-bumped ordered kg and adjusted MG

