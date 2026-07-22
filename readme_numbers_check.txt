README numeric cross-check against results/full
===============================================

Outcome: PASS

Changed individual citations already reflected in README.md:
  - late, D/D_min=1.10, epsilon=0.15, rho-averaged targeting gain:
      11.68% -> 11.67% (CSV value 11.6659754002801%)
  - late, D/D_min=1.35, epsilon=0.15, rho-averaged targeting gain:
      25.92% -> 25.93% (CSV value 25.9330624543752%)
  - drop, D/D_min=1.35, epsilon=0.15, rho-averaged targeting gain:
      36.77% -> 36.73% (CSV value 36.7285711678769%)

Other cited values checked:
  - late, rho=0.975, epsilon=0.15 online recovery:
      D/D_min=1.10: 58.03472679134% -> cited 58.0%
      D/D_min=1.20: 62.8999519172147% -> cited 62.9%
  - drop, rho=0.975, D/D_min=1.50, epsilon=0.05 recovery:
      73.0485695254057% -> cited approximately 73%
  - P2 late burst counts at D/D_min=1.50, epsilon=0.05:
      rho=0: 244.0; rho=0.975: 847.2
  - P2prime drop gap at D/D_min=1.50, epsilon=0.15:
      rho=0: 0.761526666769172% -> cited approximately 0.8%
      rho=0.975: 6.45257227400883% -> cited approximately 6.5%
  - P2prime drop gap at rho=0.975, D/D_min=1.35, epsilon=0.15:
      12.3703534764551% -> cited 12.37%
  - The referenced 20.26 check (late, D/D_min=1.35, epsilon=0.10,
    rho-averaged targeting gain) is 20.2600073806973%. This number is not an
    individual citation in the current README, so no text change was needed.

Range statements remain valid:
  - 11.7-25.9%: endpoints round from 11.6659754002801 and 25.9330624543752.
  - 15-37%: drop endpoints are 15.2220099523742 and 36.7285711678769.
  - 58-63%: tight late recovery values are 58.03472679134 and
    62.8999519172147.
  - Loose high-correlation recovery range is 97.2290256417975% to
    98.4078052941451%, supporting approximately 97-98%.

Tag reference:
  - README.md references v0.2-h1.
  - README.md contains no v0.1-h1 reference.

No README edit was required during this validation pass because the current
text was already synchronized with the regenerated full CSVs.
