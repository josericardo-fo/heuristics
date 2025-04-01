# Problema das 8 Rainhas

Este projeto implementa e compara duas técnicas heurísticas diferentes para resolver o clássico problema das 8 rainhas: Hill Climbing e Simulated Annealing.

## O Problema

O problema das 8 rainhas consiste em posicionar 8 rainhas em um tabuleiro de xadrez 8x8 de forma que nenhuma rainha possa atacar outra. Isso significa que não pode haver duas rainhas na mesma linha, coluna ou diagonal.

## Métodos Implementados

### Hill Climbing

O algoritmo Hill Climbing é uma técnica de busca local que:

- Começa com uma solução aleatória
- Gera todos os vizinhos possíveis (movendo uma rainha por vez)
- Escolhe sempre o melhor vizinho (com menos conflitos)
- Para quando não encontra vizinhos melhores que a solução atual

É uma técnica gulosa que pode ficar presa em ótimos locais, mas é muito rápida e simples de implementar.

### Simulated Annealing

O algoritmo Simulated Annealing é uma metaheurística inspirada no processo de recozimento de metais que:

- Começa com uma solução aleatória e uma "temperatura" alta
- Em cada iteração, seleciona um vizinho aleatório
- Aceita vizinhos melhores sempre
- Aceita vizinhos piores com uma probabilidade que depende da temperatura atual
- Diminui gradualmente a temperatura, reduzindo a chance de aceitar soluções piores
- Termina quando a temperatura é muito baixa ou encontra-se uma solução ótima

Esta técnica pode escapar de ótimos locais através da aceitação probabilística de movimentos "para pior".

## Resultados

Os experimentos foram executados 1000 vezes para cada algoritmo, com os seguintes resultados:

| Métrica | Hill Climbing | Simulated Annealing |
|---------|---------------|---------------------|
| Taxa de Sucesso | 14.5% | 46.9% |
| Tempo Médio de Execução | 0.00106s | 0.01478s |
| Média de Iterações | 4.17 | 901.11 |
| Conflitos Médios | 1.25 | 0.54 |
| Mínimo de Conflitos | 0 | 0 |
| Máximo de Conflitos | 4 | 2 |

## Análise dos Resultados

Os dados mostram que:

- **Simulated Annealing** tem uma taxa de sucesso significativamente maior (46.9% vs 14.5%), confirmando sua capacidade teórica de escapar de ótimos locais.
- **Simulated Annealing** executa muito mais iterações em média (901.11 vs 4.17), o que explica seu tempo de execução maior.
- **Simulated Annealing** termina com um número médio menor de conflitos (0.54 vs 1.25), mostrando maior eficácia na qualidade das soluções encontradas.

## Vantagens e Desvantagens

### Hill Climbing

**Vantagens:**

- Extremamente rápido (0.00106s em média)
- Menor número de iterações (4.17 em média)
- Implementação simples e direta

**Desvantagens:**

- Taxa de sucesso menor (14.5%)
- Mais conflitos residuais em média (1.25)
- Fica preso facilmente em ótimos locais
- Número máximo de conflitos residuais mais alto (4)

### Simulated Annealing

**Vantagens:**

- Maior taxa de sucesso (46.9%)
- Menos conflitos residuais em média (0.54)
- Menor número máximo de conflitos (2)
- Capacidade de escapar de ótimos locais

**Desvantagens:**

- Mais lento (0.01478s em média)
- Executa significativamente mais iterações (901.11 em média)
- Configuração de parâmetros (temperatura, taxa de resfriamento) pode ser desafiadora

## Conclusão

Nestes experimentos, o Simulated Annealing se mostrou superior ao Hill Climbing tanto em taxa de sucesso quanto em qualidade das soluções encontradas, confirmando sua vantagem teórica em problemas com muitos ótimos locais. Embora seja computacionalmente mais custoso, a qualidade superior das soluções justifica seu uso em cenários onde a precisão é mais importante que a velocidade.

O problema das 8 rainhas possui 92 soluções distintas, e o Simulated Annealing demonstrou maior capacidade de encontrar essas soluções devido à sua habilidade de aceitar movimentos temporariamente piores para escapar de ótimos locais.
