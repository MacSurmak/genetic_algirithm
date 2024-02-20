from main import Can

"""
density - плотность точек. 1 дает около 350 точек. 25 дает около 8000 точек
max_charge - максимальный заряд. Не гарантирует отсутствие зарядов больше, но они будет встречаться редко
charge_nsteps - минимальный шаг изменения заряда. Число частей, на которые будет разделен максимальный заряд
mkdir - сохранять ли файлы в конце. Если выставить False, то все результаты будут утеряны. Отключать только для тестов

population_size - размер популяции. Число особей в одном поколении
p_crossover - вероятность скрещивания особей
p_mutation - вероятность, что особь мутирует. В файле main можно настроить вероятность мутации для одного гена, но она сработает только если особь в принципе будет мутировать
max_generations - длительность симуляции в поколениях
tournsize - число особей для турнирного отбора. Я ставлю не больше 10% от популяции
hof_size - размер зала славы. В нем сохраняются лучшие решения, чтобы не терять их и не переоткрывать повторно
"""

can = Can(max_charge=0.01, density=25, mdfile='WT_pots.xlsx', mkdir=True, low_detail=False)
can.genetic_algorithm(population_size=250,
                      p_crossover=0.9,
                      p_mutation=0.4,
                      max_generations=1000,
                      tournsize=15,
                      hof_size=5)

# can = Can(max_charge=0.1, density=25, mdfile='WT_pots.xlsx', mkdir=True)
# can.genetic_algorithm(population_size=250,
#                       p_crossover=0.9,
#                       p_mutation=0.4,
#                       max_generations=100,
#                       tournsize=15,
#                       hof_size=10)

# For tests

# can = Can(max_charge=0.01, density=25, mdfile='WT_pots.xlsx', mkdir=True, low_detail=False)
# can.genetic_algorithm(population_size=50,
#                       p_crossover=0.9,
#                       p_mutation=0.4,
#                       max_generations=5,
#                       tournsize=5,
#                       hof_size=3)
