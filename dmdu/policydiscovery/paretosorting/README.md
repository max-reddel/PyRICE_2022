# How to use Pareto sorting

Trying to reduce the number of Pareto-optimal solutions.

--------------------------------------------------

## Sufficientarian Aggregated

```
python pareto.py \
   SUFFICIENTARIAN_AGGREGATED_3.csv \
   -o 5-8 \
   -e 1.0 0.1 1.0 1.0 \
   -m 5 \
   --output sorted_SUFFICIENTARIAN_AGGREGATED_3.csv \
   --delimiter=',' \
   --header=1 \
   --blank
```

Objectives:

    utility                             
    distance to consumption threshold   
    population consumption              
    temperature overshoot              


--------------------------------------------------

## Sufficientarian Disaggregated

```
python pareto.py \
   SUFFICIENTARIAN_DISAGGREGATED_0.csv \
   -o 5-11 \
   -e 2.0 3.0 0.2 50.0 0.3 5.0 1.0 \
   -m 5 \
   --output sorted_SUFFICIENTARIAN_DISAGGREGATED_0.csv \
   --delimiter=',' \
   --header=1 \
   --blank
```

Higher epsilon needed for #3:

```
python pareto.py \
   SUFFICIENTARIAN_DISAGGREGATED_2.csv \
   -o 5-11 \
   -e 4.0 4.0 0.5 50.0 0.5 5.0 1.0 \
   -m 5 \
   --output sorted_SUFFICIENTARIAN_DISAGGREGATED_2.csv \
   --delimiter=',' \
   --header=1 \
   --blank
```


Objectives:

    utility                             
    disutility                          
    distance to consumption threshold   
    population consumption              
    distance to damage                  
    population do damage                
    temperature overshoot               

--------------------------------------------------

## Utilitarian Disaggregated

```
python pareto.py \
   UTILITARIAN_DISAGGREGATED_0.csv \
   -o 5-7 \
   -e 1.0 1.0 1.0 \
   -m 5 \
   --output sorted_UTILITARIAN_DISAGGREGATED_0.csv \
   --delimiter=',' \
   --header=1 \
   --blank
```

Higher epsilons needed for #3:

```
python pareto.py \
   UTILITARIAN_DISAGGREGATED_3.csv \
   -o 5-7 \
   -e 1.0 3.0 1.0 \
   -m 5 \
   --output sorted_UTILITARIAN_DISAGGREGATED_3.csv \
   --delimiter=',' \
   --header=1 \
   --blank
```


Objectives:

    utility                             
    disutility                          
    temperature overshoot               


