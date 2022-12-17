# Summary

## $R^2$ figure
### Rubber
![Alt text](temp/R2_node_icnn_equal.png)

### Skin
#### Before
![Alt text](temp/R2_skin_prev.png)
#### After
![Alt text](Figures/fig_skin_R2.jpg)


## Efficiency figure
### Training on all rubber data
![Alt text](temp/Mean.png)
![Alt text](temp/Median.png)
![Alt text](temp/NODE_fits.png)

Sometimes the NODE gets stuck in local minima. Things I have tried to resolve this issue:
1. Try using median rather than mean
2. Reject initial guesses for parameters that leads to high initial loss
3. Increase/decrease learning rate (2.e-6 to 1.e-1)
4. both 2. and 3. at the same time

### Training on UT only
![Alt text](temp/Efficiency_UT_only.png)


## Psi_ii figure
![Alt text](temp/rubber_UT_d2_error.jpg)
Restricting $I_i\geq 4$
![Alt text](temp/rubber_ALL_d2.jpg)

