from main import Can

"""
vdvdensity - плотность точек. 1 дает около 350 точек. 25 дает около 8000 точек
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
can = Can(vdwdensity=25, max_charge=0.01, charge_nsteps=10000, mkdir=True)
can.genetic_algorithm(population_size=250,
                      p_crossover=0.9,
                      p_mutation=0.4,
                      max_generations=500,
                      tournsize=15,
                      hof_size=10)

# Для тестов

# can = Can(vdwdensity=10, max_charge=0.03, charge_nsteps=10000, mkdir=False)
# can.genetic_algorithm(population_size=50,
#                       p_crossover=0.9,
#                       p_mutation=0.4,
#                       max_generations=50,
#                       tournsize=5,
#                       hof_size=3)

# can = Can(vdwdensity=1, max_charge=0.1, charge_nsteps=10000, mkdir=False)
# can.genetic_algorithm(population_size=50,
#                       p_crossover=0.9,
#                       p_mutation=0.3,
#                       max_generations=10,
#                       tournsize=5,
#                       hof_size=2)
