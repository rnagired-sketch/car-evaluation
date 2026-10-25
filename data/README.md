# Data

`car.data` and `car.names` are the **Car Evaluation** dataset from the UCI Machine Learning
Repository, redistributed unchanged under its
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence.

> Bohanec, M. (1988). Car Evaluation [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5JP48

1728 rows, six categorical attributes (`buying`, `maint`, `doors`, `persons`, `lug_boot`,
`safety`) and one label (`unacc`, `acc`, `good`, `vgood`); no header, comma separated.
The tests read this copy, so they run offline. Pass it to the CLI with `--data data/car.data`,
or use `--download` to fetch a fresh copy from UCI instead.
