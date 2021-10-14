### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ a82f22e4-f35b-461a-b481-1dff43722e44
using StaticArrays

# ╔═╡ d42f842d-6c2a-40db-b0c4-e936244a9e7c
using BenchmarkTools

# ╔═╡ 99b5c818-a825-4939-849e-1cade802f63d
using Measurements

# ╔═╡ d6564250-f646-40de-9463-a956af1a5b1d
using ForwardDiff

# ╔═╡ d2f4d622-8e35-4be3-b421-39b28a748cab
using CellListMap

# ╔═╡ f049ab19-7ecf-4c65-bf6d-21352c1fe767
using FastPow

# ╔═╡ 7c792b6b-b6ee-4e30-88d5-d0b8064f2734
begin
    using Plots
    plot_font = "Computer Modern"
    default(
        fontfamily=plot_font,
        linewidth=2, framestyle=:box, label=:none, grid=false,
        size=(400,400)
    )    
end

# ╔═╡ febe8c06-b3aa-4db1-a3ea-fdc2a81bdebd
using Printf

# ╔═╡ a756dd18-fac6-4527-944e-c16d8cc4bf95
begin
    using PlutoUI
    TableOfContents()
end

# ╔═╡ a87fad48-73c1-4a08-a6f1-aae759b3c6fc
md"""
# Simulações de partículas em Julia

Leandro Martínez

Instituto de Química - Universidade de Campinas (UNICAMP)

[http://m3g.iqm.unicamp.br](http://m3g.iqm.unicamp.br) - 
[https://github.com/m3g](https://github.com/m3g)

"""

# ╔═╡ 172227c2-b27a-40db-91f4-9566c2f6cf52
md"""
# Resumo

- Elementos de uma simulação de partículas
- Performance vs. uma linguagem compilada convencional (Fortran)
- Explorando o caráter genérico das funções
- Simulações diferenciáveis e ajuste de parâmetros
- Using cell lists
- Uma implementação rápida e genérica das células ligadas
- A estratégia do Packmol
- Performance vs. NAMD
- Notas

"""

# ╔═╡ 4b484cf6-4888-4f04-b3fd-94862822b0c0
md"""
# Definindo o tipo de partícula

Definiremos, em princípio, um simples vetor em um espaço bidimensional, com coordenadas `x` e `y`.  O ponto será definido com a ajuda do pacote `StaticArrays`, que fornece uma interface conveniente para este tipo de variável, e sua aritmética. A estrutura na memória de vetores destes pontos é idêntica àquela de matrizes `N×M`, onde `N` é a dimensão do espaço (aqui 2D) e `M` é o número de pontos. Julia é "column-major", então este é o formato mais eficiente para este tipo de cálculo. 
"""

# ╔═╡ 8c444ee4-8c77-413a-bbeb-9e5ae2428876
struct Vec2D{T} <: FieldVector{2,T}
    x::T
    y::T
end

# ╔═╡ a0de01b5-779a-48c0-8d61-12b02a5f527e
md"""
Por conveniência, definiremos uma função simples que retorna um vetor aletório,
dado o intervalo de coordenadas desejado:
"""

# ╔═╡ 532eb3bc-5522-4348-afa5-5336ec6752c7
md"""
Ao definir a função acima, tomamos o cuidado de escrever um código genérico para o tipo de varíavel e dimensão, de forma que não precisamos redefir esta função mais tarde, quando realizarmos simulações com diferentes tipo de estrutura de dados. 
"""

# ╔═╡ dc5f7484-3dc3-47a7-ad4a-30f97fc14d11
md"""
## Força entre um par de partículas

Inicialmente, a função de energia será um potencial suave, que é zero em distâncias maiores que um raio de corte, e cresce quadraticamente para distâncias menores que este raio.

Se $d = ||\vec{y}-\vec{x}||$ é o módulo da da posição relativa das duas partículas, temos o potencial:

$$u(\vec{x},\vec{y},c)=
\begin{cases}
(d-c)^2 &\textrm{se} & d\leq c \\
0 & \textrm{se} & d > c \\
\end{cases}$$

para o qual a força é:

$$\vec{f_x}(\vec{x},\vec{y},c)=
\begin{cases}
2(d-c)\frac{(\vec{y}-\vec{x})}{d} &\textrm{se} & d\leq c \\
\vec{0} & \textrm{se} & d > c \\
\end{cases}$$
e
$$\vec{f_y} = -\vec{f_x}$$.


"""

# ╔═╡ ab3ff21b-bf82-4d8c-abd1-c27418956ed8
md"""
O pacote padrão `LinearAlgegra` de Julia fornece uma função `norm`, e não há razão par não usá-la (ainda que uma versão manual é simples de escrever e tería performance similar):
"""

# ╔═╡ 7a1db355-bba9-4322-9fb4-a6d7b7bdd60d
import LinearAlgebra: norm

# ╔═╡ cc49ef04-08d8-42bb-9170-9db64e275a51
md"""
As funções de energia e força são simples de entender:
"""

# ╔═╡ d00e56f2-9d3a-4dd3-80eb-3201eff39b96
md"""
E, para o caso unidimensional, com um raio de corte definido, será:
"""

# ╔═╡ b5c09cd3-6063-4a36-96cd-2d128aa11b82
const cutoff = 5.

# ╔═╡ 7719b317-e85b-4583-b401-a8614d4b2373
md"""
A função que calcula a força sobre o pair de partículas correrá de forma ingênua sobre todos os pares de partículas não-repetidos. A função `forces!` vai receber os parâmetros necessários, de forma que poderemos modificar a função mais tarde. 

Dentro da função `forces!`, a função `force_pair` vai receber quatro parâmetros: os índices das partículas e suas posições. Vamos usar os índices mais tarde. 
"""

# ╔═╡ 144119ad-ab88-4165-883a-f2fc2464a838
md"""
Vamos criar alguns pontos para explicar como a função deve ser chamada.
"""

# ╔═╡ dd9e4332-2908-40ef-b461-6b571df56cf4
md"""
A função `force_pair` será passada para a função que calcula as forças na forma de uma *closure* (função de fechamento?), que vai capturar o valor do raio de corte. A *closure* também permite que ignoremos os índices das partículas, que são parâmetros esperados pela implementação interna da função dentro de `forces!`. Por exemplo:
"""

# ╔═╡ 017cd6a8-712d-4ec5-b31f-7f1f507764f2
md"""
O terceiro argumento de `forces!` é a *closure*, que pode ser lida como:
esta é uma função que, dados `(i,j,x,y)` retorna o valor de `fₓ(x,y,cutoff)`. Então, ela é consistente com a chamada interna de `fₓ` em `forces!`, e "captura" (*closes over*) o parâmetro adicional `cutoff` que é requerido pelo cálculo. 
"""

# ╔═╡ b5206dd5-1f46-4437-929b-efd68393b12b
md"""
# Realizando uma simulação de partículas

Agora, dada a função que calcula as forças, podemos realizar uma simulação de partículas. Vamos usar um algoritmo simples de integração de Euler:

1. Calcular as forças no tempo $t$ dadas as posições $x$:
$f(t) = f(x)$

2. Atualizar as posições (usando $a = f/m$):
$x(t + dt) = x(t) + v(t)dt + a(t)dt^2/2$

3. Atualizar as velocidades:
$v(t+dt) = v(t) + a(t)dt$

4. Voltar ao passo 1.

## O código que realiza esta simulação é simples:
"""

# ╔═╡ eb5dc224-1491-11ec-1cae-d51c93cd292c
function md(
    x0::Vector{T},
    v0::Vector{T},
    mass,dt,nsteps,isave,forces!
) where T
    x = copy(x0)
    v = copy(v0)
    a = similar(x0)
    f = similar(x0)
    trajectory = [ copy(x0) ] # will store the trajectory
    for step in 1:nsteps
        # Compute forces
        forces!(f,x)
        # Accelerations
        @. a = f / mass
        # Update positions
        @. x = x + v*dt + a*dt^2/2
        # Update velocities
        @. v = v + a*dt
        # Save if required
        if mod(step,isave) == 0
            println("Saved trajectory at step: ",step)
            push!(trajectory,copy(x))
        end
    end
    return trajectory
end

# ╔═╡ 594ba1d6-2dae-4f20-9546-f52fac17c2f0
md"""
Ao usar um tipo de variável paramétrico (i. e. `Vector{T}`) garantimos que teremos um erro se as posições e velocidades não possuem o mesmo tipo de elemento. 

A notação `@.` é muito comum em Julia, e significa que estamos fazendo um cálculo elemento por elemento, sendo apenas uma notação compacta para loops. 
"""

# ╔═╡ 66c7d960-7e05-4613-84e8-2a40fe40dc3d
md"""
## Vamos rodar a simulação!
"""

# ╔═╡ e717a8d9-ccfb-4f89-b2a2-f244f108b48d
md"""
Aqui geramos um conjunto aleatório de posições e velocidades, e vamos usar massas iguais a `1.0` para todas as partículas. 
"""

# ╔═╡ eab7b195-64d5-4587-8687-48a673ab091b
md"""
## Usando condições periódicas de contorno

Nossas partículas simplesmente explodem, já que as velocidades iniciais eram aleatórias e as partículas só interagem por potenciais repulsivos. 

Podemos obter uma dinâmica mais interessante se colocarmos condições periódicas de contorno. Para isso, vamos modificar a forma de calcular as forças. 
"""

# ╔═╡ 34dc72dc-4864-47c0-b730-183f67e7aea3
md"""
## Imagens mínimas 

A função a seguir define como a imagem mínima relativa de duas partículas, de acordo com as condições periódicas de contorno, pode ser obtida (a operação se chama *wrap* em inglês), para uma caixa cúbica de lado `side`: 
"""

# ╔═╡ 02d9bf3b-708c-4293-b198-9043b334ff7e
md"""
Isto permite que calculemos a força usando:
"""

# ╔═╡ 0a5282ee-c88a-4bcc-aca2-477f28e9e04d
md"""
Noss caixa terá lado `100`:
"""

# ╔═╡ b2a4a505-47ff-40bb-9a6d-a08d91c53217
const side = 100.

# ╔═╡ fcff6973-012a-40fc-a618-f6262266287a
md"""
Para executar a simulação com as novas forças periódicas, usaremos a mesma função `md`, apenas sendo necessário passar a nova função `fₓ` na definição de uma *closure*: 
"""

# ╔═╡ 14867ffd-cde5-43f8-8399-01169ee29b73
md"""
Um detalhe relevante é que a função `fₓ` tem o mesmo nome. Como ela recebe o parâmetro `side` que não está presente na primeira definição da função, o múlitplo-despacho de Julia se encarregará de escolher o método correto. 
"""

# ╔═╡ c4798182-de75-4b59-8be7-f7cf1051364d
md"""
Ao fazer a animação da trajetória, estamos envolvendo as coordenadas nas condições periódicas (*wrapping*): 
"""

# ╔═╡ 22fb0386-d4fa-47b9-ac31-decf2731cbc1
md"""
## Performance
"""

# ╔═╡ 8e23a3ea-3039-4a5f-b37f-c4710153938e
md"""
A perforamnance em Julia pode ser medida com a macro `@time` ou com as macros do pacote `BenchmarkTools`. A compilação das funções ocorre quando um método é chamado pela primeira vez, e `BenchmarkTools` vai automaticamente descontar o tempo de compilação e fazer uma amostragem estatisticamente mais relevante.
"""

# ╔═╡ 2a3e7257-63ad-4761-beda-cec18b91f99c
md"""
Um tempo da ordem de `200ms` e alocações de `200kiB` não parecem ruins, mas também não significam nada isoladamente. O interessante é que este código é similar em performance a códigos em linguagens compiladas convencionais, como o Fortran, como mostra [este benchmark](https://github.com/m3g/2021_FortranCon/tree/main/benchmark_vs_fortran). 

"""

# ╔═╡ 49b1f040-929a-4238-acd9-6554757b592c
md"""
# Explorando implementações genéricas

## Rodando uma simulação em 3D

Não precisamos fazer muito para rodar a simulação em três dimensões. Só é necessário definir nossos vetores em 3D:
"""

# ╔═╡ 26d5c6e9-a903-4792-a0e0-dec1a2e86a01
struct Vec3D{T} <: FieldVector{3,T}
    x::T
    y::T
    z::T
end

# ╔═╡ 2aef27bd-dea6-4a93-9d0f-b9249c9dd2cd
md"""
Com isto, podemos rodar nossa simulação em 3D:
"""

# ╔═╡ b6dcb9a3-59e3-4eae-9399-fb072c704f1a
md"""
## Propagação automática de erros 

Realizer uma simulação com dimensão variável não é a aplicação mais interessante nem a mais útil da programação genérica. Podemos, por exemplo, propagar os erros nas posições das partículas, definindo um tipo de partícula que carrega consigo tanto a posição como sua incerteza acumulada. 

Um pequeno exemplo de como isto pode ser definido será mostrado aqui. Primeiro, criamos um tipo de variável que contém tanto a posição como sua incerteza:
"""

# ╔═╡ e4657169-1bb2-4d4a-ac9d-adc80499d07d
struct MyMeasurement{T}
    x::T
    Δx::T
end

# ╔═╡ 5d395353-5681-4780-983e-902fdb89eaf2
md"""
Vamos definir uma forma customizada de escrever esta variável na tela, para que fique mais bonita:
"""

# ╔═╡ e9376a4b-3d60-42eb-8681-cd2bcec13fe8
Base.show(io::IO,m::MyMeasurement) = println(io," $(m.x) ± $(m.Δx)")

# ╔═╡ c5cdd71f-5ded-482f-9205-c13f01a14d0b
m = MyMeasurement(1.0,0.1)

# ╔═╡ a0993a1f-60a6-45b5-815e-676c49a9f049
md"""
Agora, definimos a aritmética associada a esta variável. Por exemplo, a soma de duas variáveis do tipo `MyMeasurment` soma as incertezas. Mas a subtração delas também soma as incertezas. As outras incertezas são propagadas linearmente, de acordo com a primeira derivada da operação relativamente aos valores das variáveis:
"""

# ╔═╡ 4e7f8db4-b5cc-4a3e-9fa7-e62d8f2a36ac
begin
    import Base: -, +, *, /, ^, sqrt
    +(m1::MyMeasurement,m2::MyMeasurement) = MyMeasurement(m1.x+m2.x,m1.Δx+m2.Δx)
    -(m1::MyMeasurement,m2::MyMeasurement) = MyMeasurement(m1.x-m2.x,m1.Δx+m2.Δx)
    *(α,m::MyMeasurement) = MyMeasurement(α*m.x,sign(α)*α*m.Δx)
    *(m::MyMeasurement,α) = α*m
    /(m::MyMeasurement,α) = inv(α)*m
    sqrt(m::MyMeasurement{T}) where T = MyMeasurement(sqrt(m.x),inv(2*sqrt(m.x))*m.Δx)
    ^(m::MyMeasurement{T},n) where T = MyMeasurement{T}(m.x^n,n*m.x^(n-1)*m.Δx)
end

# ╔═╡ f87e4036-8f82-41c7-90c1-daa5f677488d
function random_point(::Type{PointType},range) where PointType 
    dim = length(PointType)
    T = eltype(PointType)
    p = PointType(
        range[begin] + rand(T)*(range[end]-range[begin]) for _ in 1:dim
    )
    return p
end

# ╔═╡ df33b999-4a42-4133-bf59-5a65240790cf
function energy(x::T,y::T,cutoff) where T
    Δv = y - x
    d = norm(Δv)
    if d > cutoff
        energy = zero(T)
    else
        energy = (d - cutoff)^2
    end
    return energy
end

# ╔═╡ 0f52365d-34f4-46ed-923e-3ea31c6db0ca
function fₓ(x::T,y::T,cutoff) where T
    Δv = y - x
    d = norm(Δv)
    if d > cutoff
        fₓ = zero(T)
    else
        fₓ = 2*(d - cutoff)*(Δv/d)
    end
    return fₓ
end

# ╔═╡ f58769a6-a656-42a3-8bc6-c204d4cfd897
function forces!(f::Vector{T},x::Vector{T},fₓ::F) where {T,F}
    fill!(f,zero(T))
    n = length(x)
    for i in 1:n-1
        for j in i+1:n
            fᵢ = fₓ(i,j,x[i],x[j])
            f[i] += fᵢ 
            f[j] -= fᵢ
        end
    end
    return f
end

# ╔═╡ beeb3335-5c49-47de-a1d3-3eef5f9479f1
function wrap(x,side)
    x = rem(x,side)
    if x >= side/2
        x -= side
    elseif x < -side/2
        x += side
    end
    return x
end

# ╔═╡ 0967b90d-ac88-476d-a57a-7c38dfa82204
function fₓ(x::T,y::T,cutoff,side) where T
    Δv = wrap.(y - x, side)
    d = norm(Δv)
    if d > cutoff
        fₓ = zero(T)
    else
        fₓ = 2*(d - cutoff)*(Δv/d)
    end
    return fₓ
end

# ╔═╡ 36da3e92-000c-4d4b-9abf-4cd588b3a354
md"""
Com estas definições, podemos aogra operar sobre estas variáveis do tipo `MyMeasurement` propagando as incertezas automaticamente:
"""

# ╔═╡ 70eb2e0a-a5c8-4975-8f6c-589035bea29c
sqrt((2*(m + 4*m)^2/3))

# ╔═╡ b4646a29-3efd-4bd1-bffc-3575559de937
md"""
E podemos também definir vetores 2D (ou 3D) com valores e incertezas, sem ser necessário modificar nossa definição de vetores:
"""

# ╔═╡ d32743c0-fc80-406f-83c5-4528e439589a
x = Vec2D(MyMeasurement(1.0,0.1),MyMeasurement(2.0,0.2))

# ╔═╡ 98478246-5940-4828-a8f1-9c9fa990676d
md"""
As operações sobre estes vetores, agora, propagam as incertezas também:
"""

# ╔═╡ 310f247e-3fe8-4621-ae0b-b5ee38d2ee89
2*x .+ sqrt.(x)

# ╔═╡ 8267220a-f06e-4761-b310-00f8ba44e4b1
md"""
Prpagar as incertezas em cenários mais gerais requer a definição de outras regras de progração (de derivação). Ao mesmo tempo, queremos considerar a correlação entre as variáveis, o que faz das regras propagação mais complicadas e caras computacionalmente. 

Felizmente, alguns pacotes foram criados para fazer esta propagação de erros em casos gerais. Aqui, usaremos o pacote `Measurements`: 
"""

# ╔═╡ ce916139-221a-462e-877f-88212663c05e
md"""
### Using `Measurements`
"""

# ╔═╡ 8e2903be-4975-4e14-84ed-6e712f47fe47
md"""
Usando `Measurements` não precisamos modificar nada nos nossos códigos anteriores. Apenas precisamos redefinir o conteúdo de nossas posições e velocidades, que agoram carregarão as incertezas de cada coordenada:
"""

# ╔═╡ 418f31bb-81d5-459b-b402-4fd4e3f4ab27
md"""
Vamos redefinir apenas o que signfica gerar coordenadas iniciais aleatórias, agora com incertezas:
"""

# ╔═╡ 05402cbd-78c6-4234-8680-c351c8c37778
function random_point(::Type{Vec2D{Measurement{T}}},range,Δ) where T    
    p = Vec2D(
        range[begin] + rand(T)*(range[end]-range[begin]) ± rand()*Δ,
        range[begin] + rand(T)*(range[end]-range[begin]) ± rand()*Δ
    )
    return p
end

# ╔═╡ 356ac5a4-c94e-42cb-a085-0198b29c7e52
x0 = [ random_point(Vec2D{Float64},(0,100)) for _ in 1:100] 

# ╔═╡ d23b4a92-055e-4ed7-bd46-8a3c59312993
f = similar(x0)

# ╔═╡ e6e29d1e-9a93-49db-a358-6b66f0bc3433
forces!(
    f,
    x0, 
    (i,j,x,y) -> fₓ(x,y,cutoff) # closure
) 

# ╔═╡ 3755a4f3-1842-4de2-965e-d294c06c54c7
trajectory = md((
    x0 = [random_point(Vec2D{Float64},(-50,50)) for _ in 1:100 ], 
    v0 = [random_point(Vec2D{Float64},(-1,1)) for _ in 1:100 ], 
    mass = [ 1.0 for _ in 1:100 ],
    dt = 0.1,
    nsteps = 1000,
    isave = 10,
    forces! = (f,x) -> forces!(f,x, (i,j,p1,p2) -> fₓ(p1,p2,cutoff))
)...)

# ╔═╡ 985b4ffb-7964-4b50-8c2f-e5f45f352500
trajectory_periodic = md((
    x0 = [random_point(Vec2D{Float64},(-50,50)) for _ in 1:100 ], 
    v0 = [random_point(Vec2D{Float64},(-1,1)) for _ in 1:100 ], 
    mass = [ 10.0 for _ in 1:100 ],
    dt = 0.1,
    nsteps = 1000,
    isave = 10,
    forces! = (f,x) -> forces!(f,x,(i,j,p1,p2) -> fₓ(p1,p2,cutoff,side))
)...)

# ╔═╡ 1ad401b5-20b2-489b-b2aa-92f729b1d725
@benchmark md($(
    x0 = [random_point(Vec2D{Float64},-50:50) for _ in 1:100 ], 
    v0 = [random_point(Vec2D{Float64},-1:1) for _ in 1:100 ], 
    mass = [ 1.0 for _ in 1:100 ],
    dt = 0.1,
    nsteps = 1000,
    isave = 10,
    forces! = (f,x) -> forces!(f,x, (i,j,p1,p2) -> fₓ(p1,p2,cutoff,side))
)...)

# ╔═╡ 0546ee2d-b62d-4c7a-8172-ba87b3c1aea4
trajectory_periodic_3D = md((
    x0 = [random_point(Vec3D{Float64},-50:50) for _ in 1:100 ], 
    v0 = [random_point(Vec3D{Float64},-1:1) for _ in 1:100 ], 
    mass = [ 1.0 for _ in 1:100 ],
    dt = 0.1,
    nsteps = 1000,
    isave = 10,
    forces! = (f,x) -> forces!(f,x,(i,j,p1,p2) -> fₓ(p1,p2,cutoff,side))
)...)

# ╔═╡ 4e97f24c-c237-4117-bc57-e4e88c8fb8d2
md"""
Which generates random points carrying an initial uncertainty we defined:
"""

# ╔═╡ b31da90d-7165-42de-b18d-90584affea03
random_point(Vec2D{Measurement{Float64}},(-50,50),1e-5)

# ╔═╡ 5f37640b-ffd9-4877-a78c-a699b2671919
md"""
Dadas estas novas coordenadas e velocidades, o mesmo código de simulação pod eser usado, mas agora propagando as incertezas automaticamente: 
"""

# ╔═╡ 1d6eedfd-d013-4557-9cf2-103f8fb7b72a
md"""
A trajetória, claro, é a mesma (exceto que calculamos menos passos, porque propagar os erros é caro computacionalmente):
"""

# ╔═╡ c003a61d-a434-4d7b-9214-5b52aa044248
md"""
Mas temos, agora, uma estimativa do erro de cada posição, propagado ao longo do tempo desde a incerteza inicial nas medidas:
"""

# ╔═╡ 63eb391f-0238-434a-bc3a-2fa8ed41448e
md"""
### Movimento planetário

Esta propagação de incertezas é provavelmente mais interessante de ver em um movimento planetário:
"""

# ╔═╡ 7b9bb0fd-34a5-42e1-bc35-7259447b73d0
function gravitational_force(i,j,x,y,mass)
    G = 0.00049823382528 # MKm³ / (10²⁴kg days²)
    dr = y - x
    r = norm(dr)
    return G*mass[i]*mass[j]*dr/r^3
end

# ╔═╡ 6a4e0e2e-75c5-4cab-987d-3d6b62f9bb06
md"""
Note que agora precisamos dos índices das partículas na função, porque queremos passar também a informação sobre suas massas, que são diferentes. 

Um conjunto inicial de posições e velocidades de planetas é algo que precisamos obter [experimentalmente](https://nssdc.gsfc.nasa.gov/planetary/factsheet/). Aqui, a distância está mediad em $10^6$ km, e o tempo está medido em dias. Então, a velocidade está medida em MKm por dia. 

A incerteza inicial nas posições será tomada como o diâmetro de cada planeta. Neste exemplo não introduziremos incertezas sobre as velocidades. 
"""

# ╔═╡ c91862dd-498a-4712-8e3d-b77e088cd470
planets_x0 = [
    Vec2D(  0.0 ±  1.39    , 0. ±  1.39    ), # "Sun"
    Vec2D( 57.9 ±  4.879e-3, 0. ±  4.879e-3), # "Mercury"
    Vec2D(108.2 ± 12.104e-3, 0. ± 12.104e-3), # "Venus"
    Vec2D(149.6 ± 12.756e-3, 0. ± 12.756e-3), # "Earth"
    Vec2D(227.9 ±  6.792e-3, 0. ±  6.792e-3), # "Mars"
]

# ╔═╡ a08d6e6d-ddc4-40aa-b7c4-93ea03191415
planets_v0 = [
    Vec2D(0. ± 0.,   0.0 ± 0.), # "Sun"
    Vec2D(0. ± 0.,  4.10 ± 0.), # "Mercury"
    Vec2D(0. ± 0.,  3.02 ± 0.), # "Venus"
    Vec2D(0. ± 0.,  2.57 ± 0.), # "Earth"
    Vec2D(0. ± 0.,  2.08 ± 0.)  # "Mars"  
]

# ╔═╡ a356e2cc-1cb1-457a-986c-998cf1efe008
md"""
As massas estão medidas em $10^{24}$ kg:
"""

# ╔═╡ 57141f7c-9261-4dc5-98e4-b136a15f86fc
const masses = [ 1.99e6, 0.330, 4.87, 5.97, 0.642 ]

# ╔═╡ 055e32d7-073c-40db-a267-750636b9f786
md"""
Vamos ver os planetas orbitando o sol:
"""

# ╔═╡ aaa97ce4-a5ff-4332-89a2-843cee2e5b6d
trajectory_planets = md((
    x0 = planets_x0, 
    v0 = planets_v0, 
    mass = masses,
    dt = 1, # days
    nsteps = 2*365, # two Earth years
    isave = 1, # save every day
    forces! = (f,x) -> forces!(
        f,x, (i,j,p1,p2) -> gravitational_force(i,j,p1,p2,masses)
    )
)...)

# ╔═╡ 93697e4d-369b-48e9-8b28-a0ff58604d02
md"""
Se você se perguntou porque os erros oscilam, é porque as trajetórias são periódicas. Quando todas as trajetórias inciadas no intervalo de posições associado às incertezas se cruzam, a posição é independente da posição inicial, portanto a incerteza é nula. Em outras palavras, a derivada da posição é zero em função da posição inicial, portanto a incerteza propagada linearmente é nula.  
"""

# ╔═╡ c4344e64-aa22-4328-a97a-71e44bcd289f
md"""
Uma coisa que não está bonita, no entanto, é que em dois anos a Terra não completou duas revoluções completas em torno do Sol. Alguma coisa está errada com nossos dados. Podemos melhorar isso?
"""

# ╔═╡ 827bda6f-87d4-4d36-8d89-f144f4595240
md"""
## Podemos derivar tudo!

É supreendente (ao menos para mim) que nossa simulação é completamente diferenciável. Isto quer dizer que podemos ajustar os parâmetros da simulação, e os dados, usando algoritmos de otimização que requerem derivadas analíticas. 

Aqui, especulamos que o que estava errado com nossos dados era a posição inicial da Terra. Isso fez com que a órbita fosse um pouco mais lenta do que deveria ser. 

Vamos definir, então, uma função objetivo que retorna o deslocament da Terra relativamente a sua posição inicial (no primeiro dia) depois de 2 anos. Nosso objetivo é que a Terra volte à sua posição inicial. 
"""

# ╔═╡ 1ff4077a-4742-4c5e-a8d6-c4699469a683
md"""
Primeiro, definimos uma função que exeuta uma simulação de *dois anos* da órbita da Terra. A simulação se inicia com uma posição `x` para a Terra, que é um parâmetro desta função. Vamos ser cuidadosos em definir todas as coordenadas com o mesmo tipo de variável que a o parâmetro `x` de entrada, de forma que a função permaneça genérica e os tipos todos consistentes:  
"""

# ╔═╡ 4a75498d-8f4e-406f-8b01-f6a5f153919f
function earth_orbit(x::T=149.6,nsteps=2*3650,isave=20) where T
    x0 = [
        Vec2D( zero(T), zero(T)), # "Sun"
        Vec2D(       x, zero(T))  # "Earth"
    ]
    v0 = [ 
        Vec2D( zero(T), zero(T)), # "Sun"
        Vec2D( zero(T), 2.57*one(T)), # "Earth"
    ]
    masses = [ 1.99e6, 5.97 ]
    trajectory = md((
        x0 = x0, 
        v0 = v0, 
        mass = masses,
        dt = 0.1, # days
        nsteps = nsteps, # one Earth year
        isave = isave, 
        forces! = (f,x) -> forces!(f,x, 
            (i,j,p1,p2) -> gravitational_force(i,j,p1,p2,masses)
        )
    )...)
    return trajectory
end

# ╔═╡ 3ae783ce-d06e-4cc2-b8a3-94512e8f1490
md"""
Agora, definimos nossa função objetivo, que é o módulo do vetor diferença entre a posição inicial e a posição final da Terra (queremos que ela volte para a posição inicial):
"""

# ╔═╡ 13e7da81-8581-4f32-9fdb-2599dd36a12c
function error_in_orbit(x::T=149.6) where T
	nsteps = 2*3650
    traj = earth_orbit(x,nsteps,nsteps) # Save last point only
    return norm(traj[end][2]-[x,0.])
end

# ╔═╡ 4870b1f3-3134-4ddc-a59d-fa806b456a23
md"""
Podemos ver que com os nossos dados atuais o resultado tem um erro significativo:
"""

# ╔═╡ fda6171c-9675-4f2e-b226-7ccf100529cd
error_in_orbit()

# ╔═╡ a862f8a3-0131-4644-bc90-246bf3120790
md"""
Queremos minimizar este erro, e resulta que nossa simulação é completamente diferenciável. Vamos usar o pacote `ForwarDiff` de diferenciação automática:
"""

# ╔═╡ eee3ac4b-4ddb-4699-b6e6-f0ffcc562c07
md"""
Este pacote pode ser usado sem modificações para calcualr as derivadas do erro da órbita relativamente à posição inicial `x` da Terra:
"""

# ╔═╡ 107aec28-ecb5-4007-95e5-25d0a7f0c465
ForwardDiff.derivative(error_in_orbit,149.6)

# ╔═╡ 1394e4c6-c371-47c0-8ca8-f0830d63d8ec
md"""
Para minimizar o erro, escreveremos um simples algorítmo de descida. Muitos pacotes fornecem algoritmos mais sofisticados de otimização, mas aqui vamos manter o código simples também para ilustrar que é possível implementar o algoritmo em Julia:
"""

# ╔═╡ 535716e6-9c1c-4324-a4cd-b1214df3c01d
function gradient_descent(x,f,g,tol,maxtrial)
    itrial = 0
    step = 1.0
    fx = f(x)
    gx = g(x)
    while (abs(gx) > tol) && (itrial < maxtrial) && (step > 1e-10)
        xtrial = x - gx*step
        ftrial = f(xtrial)
        if ftrial > fx
            step = step / 2
        else
            x = xtrial
            fx = ftrial
            gx = g(x)
            step = step * 2
        end
        itrial += 1
    end 
    return x, gx, itrial
end

# ╔═╡ b8edfb4e-6780-4ce7-94c1-4073ff7fa832
md"""
A derivada do erro pode ser calculada usando a função `error_in_orbit` na forma de uma *closure*:
"""

# ╔═╡ b8320f78-323c-49a9-a9f9-2748d19ecb35
error_derivative(x) = ForwardDiff.derivative(error_in_orbit,x)

# ╔═╡ 92737d73-676c-4c96-a321-831ecaf37690
md"""
Agora podemos chamar a função `gradient_descent` diretamente:
"""

# ╔═╡ 931a9c5f-8f91-4e88-956b-50c0efc9c58b
best_x0 = gradient_descent(149.6,error_in_orbit,error_derivative,1e-4,1000)

# ╔═╡ b5b96082-efde-464f-bcd4-f2e0a84befcd
md"""
O resultado é razoável: o erro na órbita diminuiu significativamente: 
"""

# ╔═╡ 7658a32c-d3da-4ec9-9d96-0d30bb18f08c
error_in_orbit(best_x0[1])

# ╔═╡ e61981d5-5448-45e9-81dc-320ac87ba813
md"""
Vamos como mudou nossa trajetória com as novas condições iniciais:
"""

# ╔═╡ 31e1bb51-c531-4c4a-8634-5caafb7e9e51
earth_traj_0 = earth_orbit(149.6)

# ╔═╡ b0b81da4-6788-45c4-b618-188a02b5e09c
earth_traj_best = earth_orbit(best_x0[1])

# ╔═╡ 47c205c3-ceae-4e12-9ade-753df1608deb
md"""
O ponto azul escuro é a trajetória corrigida, e o azul claro corresponde à trajetória original. Fomos capazes de otimizar *o ponto inicial* da trajetória com um método baseado no gradiente. Este conceito pode ser usado para ajustar parâmetros de simulações de vários tipos (simulações de partículas ou equações diferenciais em geral).
"""

# ╔═╡ 826693ff-9a9b-46b1-aeb3-767a5e6f9441
md"""
# Acelerando com CellListMap.jl
"""

# ╔═╡ d231842d-9b7a-4711-b71b-5d54041ebc1f
md"""
[`CellListMap.jl`](https://m3g.github.io/CellListMap.jl/stable/) é um pacote que implemnta [células ligadas](https://en.wikipedia.org/wiki/Cell_lists) de forma eficiente. Células ligadas são uma forma prática de reduzir o custo do cálculo de propriedades que dependem de distâncias entre partículas dentro de um raio de corte (interações de curto alcance ou listas de vizinhos). O pacote tem uma interface que premite calcular qualquer propriedade dependente da distância, como energias potenciais, forças, funções de distribuição, etc. Condições periódicas de contorno triclinicas (gerais) podem ser usadas, em duas ou três dimensões. 

O algorítmo de células ligadas mais simples é bastante simples. Muitas otimizações podem ser feitas, no entanto, na construção das listas e no trato das condições periódicas de contorno, minimizando o número de distâncias desnecessárias calculadas. O cálculo também pode ser paralelizado.
"""

# ╔═╡ 53cedd26-3742-4c23-a8b8-8a1f2bdfa135
md"""
## O algoritmo ingênuo é muito lento O(n²)
"""

# ╔═╡ 889f837d-2e26-4261-b276-5fd91efdda6a
md"""
Com ~1k partículas, o número de pares de partículas já é da ordem de centenas de milhares. O algoritmo ingênuo O(n²) já lento demais. Simulações típicas envolvem dezenas ou centenas de milhares de partículas. 
"""

# ╔═╡ 670a01e3-82f8-4c7f-8577-852081d91ed7
md"""
Para começar, vamos simular 1000 partículas: 
"""

# ╔═╡ fce1e3e0-cdf7-453f-b913-964c10fa85a6
const n_large = 1000

# ╔═╡ 69a92ac6-833c-4605-b3d0-9400e4572886
md"""
Nosso sistema anterior tinha 100 partículas em um quadrado de lado 100. Vamos manter a densidade constante: 
"""

# ╔═╡ 542a9ef5-d9ee-49bd-9d31-61e28b80b5cb
const box_side = sqrt(n_large / (100/100^2))

# ╔═╡ 8bada25c-b586-42b4-851d-232ccca8a456
md"""
Só precisamos gerar novas coordenadas e rodar a simulação:
"""

# ╔═╡ 7600c6dc-769e-4c77-8526-281a1bcec079
x0_large = [ Vec2D(box_side*rand(),box_side*rand()) for _ in 1:n_large ] 

# ╔═╡ 29dbc47b-3697-4fdf-8f34-890ab4d0cdae
t_naive = @elapsed trajectory_periodic_large = md((
    x0 = x0_large, 
    v0 = [random_point(Vec2D{Float64},(-1,1)) for _ in 1:n_large ], 
    mass = [ 10.0 for _ in 1:n_large ],
    dt = 0.1,
    nsteps = 1000,
    isave = 10,
    forces! = (f,x) -> forces!(f,x,(i,j,p1,p2) -> fₓ(p1,p2,cutoff,box_side))
)...)

# ╔═╡ 0ee7fc18-f41f-4179-a75e-1e1d56b2db29
md""" 
Tempo de execução do algoritmo ingênuo: $t_naive seconds
"""

# ╔═╡ 0d0374ed-5150-40e6-b5a4-9a344b6ca47a
md"""
## Usando células ligadas
"""

# ╔═╡ f7cf613e-be9d-4f62-a778-cc4375eb99df
md"""
Nas células ligadas, as partículas são classificadas em células antes de qualquer cálculo de distâncias. As distâncias são calculadas apenas para partículas de células vizinhas. Se o tamanho das células é muito menor que o tamanho total do sistema, o número de distâncias calculado é drasticamente reduzido. 
"""

# ╔═╡ 5be87c6f-5c31-4d14-a8cb-4e63ef39d538
begin
    
function cell_list_picture()
    
    function square(c,side)
          x = [ c[1]-side/2, c[1]+side/2, c[1]+side/2, c[1]-side/2, c[1]-side/2]  
          y = [ c[2]-side/2, c[2]-side/2, c[2]+side/2, c[2]+side/2, c[2]-side/2]
          return x, y
    end
    
    plt = plot()
    
    x,y=square([5,5],2)
    plot!(
        plt,x,y,seriestype=[:shape],
        linewidth=2,fillalpha=0.05,color="green",label=""
    )
    
    x,y=square([5,5],6)
    plot!(
        plt,x,y,seriestype=[:shape],
          linewidth=2,fillalpha=0.05,color="orange",label=""
    )
    
    lines = collect(2:2:8)
    vline!(plt,lines,color="gray",label="",style=:dash)
    hline!(plt,lines,color="gray",label="",style=:dash)
    
    px = [ 0.1 + 9.8*rand() for i in 1:100 ]
    py = [ 0.1 + 9.8*rand() for i in 1:100 ]
    scatter!(plt,px,py,label="",alpha=0.20,color="blue")
    
    fontsize=8
    annotate!(plt,3,3,text("(i-1,j-1)",fontsize,:Courier))
    annotate!(plt,5,3,text("(i-1,j)",fontsize,:Courier))
    annotate!(plt,7,3,text("(i-1,j+1)",fontsize,:Courier))
    
    annotate!(plt,3,5,text("(i,j-1)",fontsize,:Courier))
    annotate!(plt,5,5,text("(i,j)",fontsize,:Courier))
    annotate!(plt,7,5,text("(i,j+1)",fontsize,:Courier))
    
    annotate!(plt,3,7,text("(i+1,j-1)",fontsize,:Courier))
    annotate!(plt,5,7,text("(i+1,j)",fontsize,:Courier))
    annotate!(plt,7,7,text("(i+1,j+1)",fontsize,:Courier))
    
    plot!(
        plt,size=(400,400), 
        xlim=(1.3,8.7),xticks=:none,
        ylim=(1.3,8.7),yticks=:none,
        framestyle=:box,
        xlabel="x",ylabel="y",grid=false
    )
    
    return plt
end

cell_list_picture()
end

# ╔═╡ 0c07edd3-c0a1-4f72-a16a-74badb7a6123
md"""
Para usar o `CellListMap.jl`, precisamos definir o sistema, fornecendo os dados do tamanho da caixa e o raio de corte:
"""

# ╔═╡ 4fc5ef4d-e072-41f7-aef9-b42730c8313c
box = Box([box_side,box_side],cutoff)

# ╔═╡ 19c5cc9d-8304-4e36-a3ea-a1151f28f71d
md"""
As partículas são então classificadas nas células. Partículas virtuais são criadas nas fronteiras para lidar com as condições periódicas de contorno e evitar ter que calcular imagens mínimas durante o cálculo das interações por pares: 
"""

# ╔═╡ 7dcadd85-2986-4e42-aa84-67128a8f666d
cl = CellList(x0_large,box)

# ╔═╡ 0b5c6ede-bceb-499a-a9a8-3c6a75ed340a
md"""
Só precisamos implementar a função que é avaliada *se* o par de partículas está dentro do raio de corte. Esta função só será chamada nesta condição. Aqui, a função vai atualizar o vetor de forças:
"""

# ╔═╡ 91b5eac1-4799-4a72-ac6a-e2b117b787d5
function fpair_cl(x,y,i,j,d2,f,box::Box)
    Δv = y - x
    d = sqrt(d2)
    fₓ = 2*(d - box.cutoff)*(Δv/d)
    f[i] += fₓ
    f[j] -= fₓ
    return f
end

# ╔═╡ 0f86ab3c-29aa-472b-8194-228c736ee940
md"""
A função que calcula as forças na nossa simulação vai, então, consistir na atualização das listas de células, seguida da chamada da função `map_pairwise!` de `CellListMap.jl`, que recebe como argumento a função que atualiza a força entre um par de partículas (`fpair_cl`, aqui), além do vetor de forças que deve ser atualizado, `f`, e as propriedades do sistema. Aqui vamos executar somente a versão serial:
"""

# ╔═╡ 0b8a2292-c0d6-44e4-b560-32d9d579a008
function forces_cl!(f::Vector{T},x,box::Box,cl::CellList,fpair::F) where {T,F}
    fill!(f,zero(T))
    cl = UpdateCellList!(x,box,cl,parallel=false)
    map_pairwise!(
        (x,y,i,j,d2,f) -> fpair(x,y,i,j,d2,f,box),
        f, box, cl, parallel=false
    )
    return f
end

# ╔═╡ d6585cca-78bf-41d1-aea3-01d9831d76cb
md"""
Agora que temos uma definição apropriada da função que calcula as forças, podemos rodar a simulação novamente:
"""

# ╔═╡ 1b7b7d48-79d2-4317-9045-5b7e7bd073e5
t_cell_lists = @elapsed trajectory_cell_lists = md((
    x0 = x0_large, 
    v0 = [random_point(Vec2D{Float64},(-1,1)) for _ in 1:n_large ], 
    mass = [ 10.0 for _ in 1:n_large ],
    dt = 0.1,
    nsteps = 1000,
    isave = 10,
    forces! = (f,x) -> forces_cl!(f,x,box,cl,fpair_cl)
)...)

# ╔═╡ 3f9dad58-294c-405c-bfc4-67855bb1e825
md""" 
Tempo de execução usando `CellListMap.jl`: $t_cell_lists segundos (na segunda execução, a compilação demora cerca de 2 segundos).
"""

# ╔═╡ 6d61b58f-b88f-48f4-8bdd-0bb1a8bc1c82
md"""
Mesmo para um sistema pequeno como este, a simulação foi acelerada significativamente (em terno de $(round(Int,t_naive/t_cell_lists)) vezes). 
"""

# ╔═╡ 76b8695e-64dc-44bc-8938-ce22c4a9e4d0
md"""
## Minimização da energia: A estratégia do Packmol
"""

# ╔═╡ 372637ff-9305-4d45-bf6e-e6531dadbd14
md"""
### O potencial de Lennard-Jones

Simulações de dinâmica moelcular geralmente envovlem o cálculo, para cada par de átomos, de um pontecial de Lennard-Jones, que tem a forma:

$$u(r) = \varepsilon\left(\frac{\sigma^{12}}{r^{12}} - 2\frac{\sigma^6}{r^6}\right)$$

As potências altas envolvidas fazem o cálculo desta função numericamente instável. Vamos tentar minimizar a energia de um conjunto de pontos gerado aleatoriamente.

Vamos definir uma função que soma a energia de todas as contribuições de pares de partículas, e usar a função `map_pairwise!` do `CellListMap.jl` para calcular a contribuição de todos os pares de partículas mais próximas que um dado raio de corte.  
"""

# ╔═╡ 3b2c08a6-b27e-49be-a0b0-e5cb3d5546e0
md"""
O pacote `FastPow` expande as potências altas em produtos de potenciais menores, que são mais rápidas de calcular (com alguma perda de precisão, que não é importante aqui). Esta expansão poderia ser feita à mão, mas para que o código fique mais claro, e por conveniência, usaremos a macro `@fastpow`.
"""

# ╔═╡ 7280e368-c68a-48a5-91fe-93c76607c144
md"""
A função que calcula a energia associada a um par de partículas é:
"""

# ╔═╡ 755fae26-6db9-45a0-a60d-d0e9c063f8aa
function ulj_pair(r2,u,ε,σ)
    @fastpow u += ε*(σ^12/r2^6 - 2*σ^6/r2^3)
    return u
end

# ╔═╡ 9a8d8012-ba54-4d9b-8c4c-fe6358508f2a
md"""
E a função que calcula a energia total, mapeando esta função de pares para todos os pares de partículas mais próximos que o raio de corte é, então:
"""

# ╔═╡ ffbeae5f-8aec-4473-a446-5b73bd911733
function ulj(x,ε,σ,box::Box,cl::CellList)
    cl = UpdateCellList!(x,box,cl,parallel=false)
    u = map_pairwise!(
        (x,y,i,j,d2,u) -> ulj_pair(d2,u,ε,σ),
        zero(eltype(σ)), box, cl,
        parallel=false
    )
    return u
end

# ╔═╡ 3738e40f-9596-469d-aa58-a4b28d8a22f8
md"""
As funções correspondentes para calcular as forças são:
"""

# ╔═╡ 5f1054b8-2337-43c1-a086-26233e95d42b
function flj_pair!(x,y,i,j,r2,f,ε,σ)
    @fastpow ∂u∂x = 12*ε*(σ^12/r2^7 - σ^6/r2^4)*(y-x)
    f[i] -= ∂u∂x
    f[j] += ∂u∂x
    return f
end

# ╔═╡ bd719619-bdd4-4c3c-8d66-1df0f210c595
function flj!(f::Vector{T},x,ε,σ,box,cl) where T
    cl = UpdateCellList!(x,box,cl,parallel=false)
    fill!(f,zero(T))
    map_pairwise!(
        (x,y,i,j,d2,f) -> flj_pair!(x,y,i,j,d2,f,ε,σ),
        f, box, cl, 
        parallel=false
    )
    return f
end

# ╔═╡ f289955b-0239-4b8d-ba08-2edf0a7284c2
md"""
### Um sistema físico: Um gás de Neônio

Vamos simular um sistema com algum significado físico, que é mais interessante que apenas um conjunto de pontos quaisquer.

Vamos calcular a energia de um gás de Neônio com 10 mil partículas, com densidade $\sim 1$ partículas/Å³, que é aproximadamente a densidade atômica na água líquida. 

Os parâmetros de Lennard-Jones para o Neônio são:
"""

# ╔═╡ 878ab5f7-28c1-4832-9c58-cb36b360766f
const ε = 0.0441795 # kcal/mol

# ╔═╡ a0dcd888-059d-4abe-bb6b-958d2879101c
const σ = 2*1.64009 # Å

# ╔═╡ b1c5b7e5-cfbf-4d93-bd79-2924d957ae14
md"""
Definindo uma densidade adequada para estas dez mil partículas, 
o número de pares de partículas dentro do raio de corte é da ordem do número típico de uma simulação de dinâmica molecular convencional (estamos na verdade simulando um gás de Neônio em alta pressão).
"""

# ╔═╡ cd1102ac-1500-4d79-be83-72ac9180c7ce
const n_Ne = 10_000

# ╔═╡ f21604f4-e4f7-4d43-b3d9-32f429df443e
const box_side_Ne = (10_000/0.1)^(1/3)

# ╔═╡ 10826a95-16f8-416d-b8c1-0ef3347c9b20
x0_Ne = [random_point(Vec3D{Float64},(0,box_side_Ne)) for _ in 1:n_Ne ]

# ╔═╡ c46a4f97-78e4-42fd-82b3-4dc6ce99abac
md"""
Dadas as posições iniciais, o tamanho da caixa e o raio de corte típico de 12Å, podemos inicializar as células ligadas: 
"""

# ╔═╡ 7c433791-a653-4836-91e2-084355c01d90
const box_Ne = Box([box_side_Ne for _ in 1:3],12.)

# ╔═╡ 410b9da4-7848-4385-bffc-a3d9bd03cf19
const cl_Ne = CellList(x0_Ne,box_Ne)

# ╔═╡ c08fff28-520e-40af-951c-fe3c324f67e0
md"""
### Primeiro, vamos tentar minimizar a energia 
"""

# ╔═╡ eb0f9080-2523-4633-be21-3a2281a1629e
md"""
A primeira coisa que fazemos em uma simulação de dinâmica molecular é minimizar a energia do sistema para eliminar maus contatos (afastar partículas que estão muito próximas).
"""

# ╔═╡ e3cc3c77-71ad-4006-8e27-fabaa1ae9cfb
md"""
A energia obtida (após 500 passos do método do gradiente), é:
"""

# ╔═╡ 739c9a8a-13a5-4a33-a441-f5bc6cb35e82
md"""
### Empacotando os átomos
"""

# ╔═╡ 4e059cb8-6dac-450d-9f46-b3e657d9c3cf
md"""
Para empacotar os átomos, o raio de corte precisa ser da ordem dos seus raios,
em lugar do raio de corte das interações de Lennard-Jones. Então, redefinimos as listas de células com um raio de corte de σ/2. Como apenas interações de muito curto alcance serão calculadas, e a função é bem comportada, a otimização é muito rápida:
"""

# ╔═╡ 5ff9c89a-d999-4af2-8e7e-fb99d4948c36
md"""
Vamos redefinir a caixa, com o raio de corte menor:
"""

# ╔═╡ 0f2f55f6-060b-475e-bef7-eaa99da4d99f
box_pack = Box([box_side_Ne for _ in 1:3],σ/2)

# ╔═╡ 415ad590-247b-4a5d-b21e-7af4d0c17493
cl_pack = CellList(x0_Ne,box_pack)

# ╔═╡ 53c2e16e-b7f5-4df2-96f4-08402b5f8979
md"""
Já havíamos definido uma função para calcular as forças de curto alcance (`forces_cl`), mas precisamos aqui a função de "energia" associada:
"""

# ╔═╡ 16cdbc18-e846-4d0a-b7e6-87f07c0c52d9
function u_pack(x,box::Box,cl::CellList)
    cl = UpdateCellList!(x,box,cl,parallel=false)
    u = map_pairwise!(
        (x,y,i,j,d2,u) -> begin
			u += (sqrt(d2) - box.cutoff)^2 # objective function
			return u
		end,
        0., box, cl,
        parallel=false
    )
    return u
end

# ╔═╡ 79169f89-fedc-466b-8170-fff99b98e147
md"""
Dadas as funções de energia o gradiente das funções de empacotamente, podemos resolver o problema:
"""

# ╔═╡ b5e258cd-5542-4a4c-ae0f-91c2fee426db
md"""
É importante notar que a otimização do empacotamento convergiu para um minimizador global (a valor da função é zero):
"""

# ╔═╡ b0dc1c2b-82b7-488d-8074-1ef9f59a15e5
md"""
A energia, por outro lado, não é necessariamente baixa:
"""

# ╔═╡ 97b8b15b-75c7-4321-999d-b067ed2a04f9
md"""
Em duas dimensões, esta é a diferença entre um conjunto de pontos gerado aleatoriamente, e um conjunto de pontos obtido após a solução do problema de empacotamento:
"""

# ╔═╡ 591e6a9c-444c-471f-a56b-4dfbc9111989
md"""
Mesmo a energia total sendo alta, temos a garantia de que nenhum par de átomos está mais próximo que o raio de corte mínimo tolerável para as interações repulsivas de curto alcance. Assim, esta configuração é adequada para iniciar uma simulação.

`Packmol` resolve este problema de empacotamento de para moléculas de geometria complexa, e ao mesmo tempo permite ao usuário a especificação de diferentes restrições geométricas que podem ser usadas para definir o arranjo das espécies no espaço. 
"""

# ╔═╡ 1f265576-824a-4764-a738-685554068079
md"""
## Quão rápido é `CellListMap.jl`?

Uma ideia da eficiência da implementação das células ligadas em `CellListMap.jl` pode ser obtida comparando o tempo de uma simulação do gás de Neônio, com a mesma simulação feita por um pacote estabelecido de dinâmica molecular, como o [NAMD](https://www.ks.uiuc.edu/Research/namd/). 

Podemos executar a simulação usando a mesma função `md` de antes, mas com a função de energia potencial de Lennard-Jones correta:
"""

# ╔═╡ 1e099e1f-6494-419b-8517-5bded3e18aa6
md"""
# Notas

1. Muitos tipos de distribuições podem ser obtidas usando o pacote `Distributions.jl`. O caráter geral das funções em Julia permite que um pacote como este seja usado para gerar distribuições de tipos próprios, como o `Vec2D` implementado aqui. 

2. A integração do nosso código `md` é a mais simples de todas. O algoritmo padrão para integração de equações de movimento em dinâmica molecular é o `Velocity-Verlet`, e é este que está implementado no código que compara a eficiência dos `CellListMap.jl` em relação ao NAMD. Muitas outras alternativas estão disponíveis no pacote `DifferentialEquations.jl`.

3. A propagação de incertezas e a diferenciabilidade da simulação de partículas precisa ser vista com um pouco de cautela. Estes sistemas são tipicamente caóticos, portanto as incetezas aumentam de forma exponencial, e as derivadas são muito instáveis. Um blog post interessante sobre a sensibilidade destes cálculos pode está disponível [aqui](https://frankschae.github.io/post/shadowing/). Novamente, o pacote `DifferentialEquations.jl` possui ferramentas mais adequadas para lidar com a otimização de parâmetros das simulações nessas circunstâncias. 

"""

# ╔═╡ 10c86547-d4f4-4c3f-8906-ac18ce93f3b6
md"""
# Agradecimentos

Agradeço a [Mosè Giordano](https://giordano.github.io/) por valiosas discussões a respeito do funcionamento pacote `Measurements`, e a [Chris Foster](https://github.com/c42f) por discussões a respeito da propagação de erros nas trajetórias e outras sugestões para o notebook. Também agradeço a vários outros membros dos fórums *discourse* de Julia e Fortran que contribuiram direta ou indiretamente com este trabalho. Agradeço finalmente ao comitê organizador da FortanCon2021 pelo convite, em particular a [Milan Curcic](https://milancurcic.com) e a [Ondřej Čertík](https://github.com/certik), e ao Prof. [Mendeli Vainstein](http://www.if.ufrgs.br/posfis-antigo/index.php/br/permanentes/431-mendeli-vainstein.html) pelo convite para apresentar o seminário na UFRGS. 
"""

# ╔═╡ 2871aca3-e6b4-4a2d-868a-36562e9a274c
md"""
# Configurações do notebook
"""

# ╔═╡ 2a2e9155-1c77-46fd-8502-8431573f94d0
md"""
## Configurações padrão para os gráficos
"""

# ╔═╡ b557a646-8c3c-4dc7-8788-bf98aec8c5c0
md"""
Usamos `Printf` para formatar alguns dados.
"""

# ╔═╡ 260d5753-6cc2-4137-8a2c-8d8a47585ecf
md"""
Podemos definir esta variável como `false` e não fazer nenhum gráfico.
"""

# ╔═╡ a9981931-4cc9-4d16-a6d2-34b4071a84d7
const build_plots = true

# ╔═╡ 374f239b-6470-40ed-b068-a8ecaace4f09
build_plots && begin
    r = 0:0.1:1.2*cutoff
    plot(layout=(1,2),size=(600,300))
    plot!(r,energy.(0.,r,cutoff),xlabel="Distance",ylabel="Energy",subplot=1)
    plot!(r,fₓ.(0.,r,cutoff),xlabel="Distance",ylabel="Force",subplot=2)
end    

# ╔═╡ 43e6b146-ee35-40f1-b540-3da22b9e1b1b
build_plots && scatter([(x.x, x.y) for x in x0])

# ╔═╡ 505ef5ab-f131-4ab3-a723-795b5eb5dc0f
build_plots && @gif for (step,x) in pairs(trajectory)
    scatter([ (p.x,p.y) for p in x ], lims=(-250,250))
    annotate!(130,-210,text("step: $step",plot_font,12,:left))
end

# ╔═╡ efc586a2-0946-4dc5-ab3a-3902a811f3ad
build_plots && @gif for (step,x) in pairs(trajectory_periodic)
    scatter([ wrap.((p.x,p.y),100) for p in x ], lims=(-60,60))
    annotate!(25,-50,text("step: $step",plot_font,12,:left))
end

# ╔═╡ 4a498c18-406f-4437-b378-aa9fdc75b919
build_plots && @gif for x in trajectory_periodic_3D
    scatter([ wrap.((p.x,p.y,p.z),100) for p in x ], lims=(-60,60))
end

# ╔═╡ d87c22d1-d595-4d43-ab1c-f28d282a3485
build_plots && ( trajectory_2D_error = md((
    x0 = [random_point(Vec2D{Measurement{Float64}},(-50,50),1e-5) for _ in 1:100 ], 
    v0 = [random_point(Vec2D{Measurement{Float64}},(-1,1),1e-5) for _ in 1:100 ],
    mass = [ 1.0 for _ in 1:100 ],
    dt = 0.1,
    nsteps = 100,
    isave = 1,
    forces! = (f,x) -> forces!(f,x, (i,j,p1,p2) -> fₓ(p1,p2,cutoff,side))
)...) )

# ╔═╡ bf0a5303-f5ce-4711-b9ee-a12ce2d8a397
build_plots && @gif for x in trajectory_2D_error
    positions = [ wrap.((p.x.val,p.y.val),100) for p in x ]
    scatter(positions, lims=(-60,60))
end

# ╔═╡ e24ce081-e367-4feb-8a79-66b8654a0b3a
build_plots && @gif for x in trajectory_2D_error
    histogram(
        [ p.x.err for p in x ],
        xlabel="Uncertainty in x",ylabel="Number of points",
        bins=0:1e-4:20e-4,ylims=[0,50]
    )
end

# ╔═╡ 1067527e-76b7-4331-b3ab-efd72fb99dfc
build_plots && @gif for (step,x) in pairs(trajectory_planets)
    colors = [ :yellow, :grey, :brown, :blue, :red ]
    positions = [ (p.x.val,p.y.val) for p in x ]
    xerr = [ p.x.err for p in x ]
    yerr = [ p.y.err for p in x ] 
    scatter(positions,lims=[-250,250], markercolor=colors, xerror=xerr, yerror=yerr)
    annotate!(150,-210,text(@sprintf("%5i days",step),plot_font,12))
end

# ╔═╡ 4cef9cea-1e84-42b9-bff6-b9a8b3bfe8da
build_plots && @gif for step in eachindex(earth_traj_best)
    colors = [ :yellow, :blue ]
    positions0 = [ (p.x,p.y) for p in earth_traj_0[step] ] 
    positions_best = [ (p.x,p.y) for p in earth_traj_best[step] ]
    scatter(positions0,lims=[-250,250], markercolor=colors, alpha=0.5)
    scatter!(positions_best,lims=[-250,250], markercolor=colors)
    scatter!(
        (earth_traj_best[1][2].x,earth_traj_best[1][2].y),
        markercolor=:white,alpha=0.5,
        markersize=10
    )
    annotate!(150,-210,text(@sprintf("%5i days",2*step),plot_font,12))
end

# ╔═╡ e5b557d7-0952-4409-ae4c-a0c8ce736e03
build_plots && @gif for (step,x) in pairs(trajectory_periodic_large)
    scatter(
        [ wrap.((p.x,p.y),box_side) for p in x ], 
        lims=(-1.1*box_side/2,1.1*box_side/2)
    )
end

# ╔═╡ 30d2f39e-5df2-4f38-8032-e5f8492ba335
build_plots && @gif for (step,x) in pairs(trajectory_cell_lists)
    scatter(
        [ wrap.((p.x,p.y),box_side) for p in x ], 
        lims=(-1.1*box_side/2,1.1*box_side/2)
    )
end

# ╔═╡ 2634feff-7442-4d8f-b8e5-c11113136980
build_plots && begin
	r_lj = 2.8:0.05:8
	plot(
		r_lj,ulj_pair.(r_lj.^2,0.,ε,σ),
		xlabel="Distance / Å", ylabel="Potential energy / kcal/mol"
	)
end

# ╔═╡ b4154fb7-e0b0-4211-8490-8a8fe47cd2da
md"""
## Método do gradiente para vetores
"""

# ╔═╡ 8ac7b1bf-c958-4eb5-8376-f802b372e796
function gradient_descent!(x::Vector{T},f,g!;tol=1e-3,maxtrial=500) where T
    gnorm(x) = maximum(norm(v) for v in x)
    itrial = 0
    step = 1.0
    xtrial = similar(x)
    g = fill!(similar(x),zero(T))
    fx = f(x)
    g = g!(g,x)
    while (gnorm(g) > tol) && (itrial < maxtrial) 
        @. xtrial = x - step*g
        ftrial = f(xtrial)  
        if ftrial >= fx
            step = step / 2
        else
            x .= xtrial
            fx = ftrial
            g = g!(g,x)
            step = step * 2
        end
        @show itrial, step, fx, ftrial, gnorm(g)
		itrial += 1
    end 
    return x
end

# ╔═╡ 357c6621-b2b8-4f30-ba41-ffc1ae6f031b
t_min = @elapsed x_min = gradient_descent!(
    copy(x0_Ne),
    (x) -> ulj(x,ε,σ,box_Ne,cl_Ne),
    (g,x) -> -flj!(g,x,ε,σ,box_Ne,cl_Ne)
)

# ╔═╡ 58eb5b4b-76ad-4f7a-b86b-0494a857dca1
ulj(x_min,ε,σ,box_Ne,cl_Ne)

# ╔═╡ 574047fa-6626-4cd0-8317-32118129711e
md"""
O exemplo aqui é somente ilustrativo, mas mostra um comportamento comum: a energia depois da minimização é ainda muito alta. O custo computacional e a instabilidade numérica do potencial verdadeiro, em curtas distâncias, faz da energia uma função difícil de minimizar. 

**Tempo usado na mininimzação da energia: $t_min seconds**

Por causa disto [`Packmol`](http://m3g.iqm.unicamp.br/packmol) foi desenvolvido. Primeiro, resolvemos o problema de empacotar os átomos no espaço, garantindo que a menor distância entre eles seja maior que uma tolerância mínima. Aqui, isto consiste na minimização do nosso potencial suave original:
"""

# ╔═╡ 224336e2-522c-44af-b9a1-307e2ffff0f9
t_pack = @elapsed x_pack = gradient_descent!(
    copy(x0_Ne),
    (x) -> u_pack(x,box_pack,cl_pack),
    (g,x) -> -forces_cl!(g,x,box_pack,cl_pack,fpair_cl)
)

# ╔═╡ 06526edf-911a-4ecc-a350-6d932ca56cd5
md"""
**Tempo necessário para o empacotamento: $t_pack seconds**
"""

# ╔═╡ c48210e0-1a04-4f84-a4e2-f6b5d34a603d
u_pack(x_pack,box_pack,cl_pack)

# ╔═╡ 0471b987-f987-4656-b961-285e32d0a5e1
ulj(x_pack,ε,σ,box_Ne,cl_Ne)

# ╔═╡ 339487cd-8ee8-4d1d-984b-b4c5ff00bae3
t_Ne = @elapsed trajectory_Ne = md((
    x0 = x_pack, 
    v0 = [random_point(Vec3D{Float64},(-1,1)) for _ in 1:n_Ne ], 
    mass = [ 20.179 for _ in 1:n_Ne ],
    dt = 0.01,
    nsteps = 100,
    isave = 10,
    forces! = (f,x) -> flj!(f,x,ε,σ,box_Ne,cl_Ne)
)...)

# ╔═╡ 9cb29b01-7f49-4145-96d8-c8fd971fe1c8
build_plots && @gif for x in trajectory_Ne
    scatter(
      [ wrap.((p.x,p.y,p.z),box_side_Ne) for p in x ], 
      lims=(-1.1*box_side_Ne/2,1.1*box_side_Ne/2)
    )
end

# ╔═╡ ac52a71b-1138-4f1b-99c3-c174d9f09187
md"""
A simulação demorou $t_Ne seconds. 

Novamente, para entender exatamente o que isto significa, temos que fazer uma comparação apropriada. [Neste benchmark](https://github.com/m3g/2021_FortranCon/tree/main/celllistmap_vs_namd) fazemos duas simulações deste mesmo gás, com condições termodinâmicas apropriadas. O algoritmo implementado em Julia é similar ao que está descrito aqui, exceto que é feita uma termalização por reescalonamento de velocidades, e o algoritmo Velocity-Verlet de integração é usado (o que aumenta a estabilidade da simulação, mas não a performance relativamente ao método simples aqui usado). Os mesmo algoritmos são usados na simulação realizada com o NAMD. O resultado do benchmark foi obtido em um laptop com 4 processadores físicos (e 8 threads).

````
NAMD:

real    1m14,049s
user    8m59,065s
sys     0m1,130s

CellListMap:

real    1m21,054s
user    7m38,053s
sys     0m2,172s
````

Esta comparação claro que pode ser questionada: `NAMD` é um pacote de simulação de propósito geral, desenvolvido para ter eficiente em simulações massivamente paralelas. O `CellListMap.jl` é um pacote para computar qualquer propriedade dependente da distância, mas por enquanto otimizado apenas para computadores com memória compartilhada, e menos processadores.  

No entanto, o benchmark mostra que é possível escrever códigos de alta peformance em Julia, e que `CellListMap.jl` é uma ferramenta poderosa para a simlação ou cálculo de propriedades e análises customizadas de simulações de partículas. 

Uma das aplicações deste pacote consiste no cálculo de funções de distribuição de sistemas químicos de grande complexidade estruural, em [ComplexMixtures.jl](https://m3g.github.io/ComplexMixtures.jl/stable/).

"""

# ╔═╡ b5008faf-fd43-45dd-a5a1-7f51e0b4ede5
md"""
## Conteúdo
"""

# ╔═╡ 555d1f62-b95b-4377-a8e2-9e442ee7526d


# ╔═╡ f5510c1e-9b9f-49f0-bc7e-0fd8e79a5760
md"""
## Exemplo de empacotamento em 2D
"""

# ╔═╡ 5d9a40b5-4050-47d2-9855-e9b62d56e8df
side_test = 50

# ╔═╡ 7f556f7c-cdb0-4f91-a359-2f933bbc5b68
xtest = [ random_point(Vec2D{Float64},(-side_test,side_test)) for _ in 1:1000 ]

# ╔═╡ 0fc843d2-ac4f-4717-a298-92a476223112
tol_test = 2

# ╔═╡ fc7f665b-00d9-431b-a97e-d2ff7253221a
box_test = Box([side_test,side_test],tol_test)

# ╔═╡ aadf6e48-0cbf-4973-86a1-173b6648d1df
cl_test = CellList(xtest,box_test)

# ╔═╡ 452e1ea7-98be-4910-ba6b-c0881fb251b2
x_pack_test = gradient_descent!(
    copy(xtest),
    (x) -> u_pack(x,box_test,cl_test),
    (g,x) -> -forces_cl!(g,x,box_test,cl_test,fpair_cl)
)

# ╔═╡ d9f254dc-ae4a-40b3-b682-f8a501e10a2d
build_plots && begin
	plot(layout=(1,2))
	scatter!([ wrap.(Tuple(p),side_test) for p in xtest ],subplot=1)
	scatter!([ wrap.(Tuple(p),side_test) for p in x_pack_test ],subplot=2)
	plot!(lims=(-1.1*side_test/2,1.1*side_test/2),aspect_ratio=1,size=(800,400))
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CellListMap = "69e1c6dd-3888-40e6-b3c8-31ac5f578864"
FastPow = "c0e83750-1142-43a8-81cf-6c956b72b4d1"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[compat]
BenchmarkTools = "~1.2.0"
CellListMap = "~0.5.20"
FastPow = "~0.1.0"
ForwardDiff = "~0.10.19"
Measurements = "~2.6.0"
Plots = "~1.22.2"
PlutoUI = "~0.7.11"
StaticArrays = "~1.2.12"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "61adeb0823084487000600ef8b1c00cc2474cd47"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.2.0"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[CellListMap]]
deps = ["DocStringExtensions", "LinearAlgebra", "Parameters", "ProgressMeter", "Random", "Setfield", "StaticArrays"]
git-tree-sha1 = "6307da1063024d02438f460b98a7539fc6270298"
uuid = "69e1c6dd-3888-40e6-b3c8-31ac5f578864"
version = "0.5.20"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bd4afa1fdeec0c8b89dad3c6e92bc6e3b0fec9ce"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.6.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "31d0151f5716b655421d9d75b7fa74cc4e744df2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.39.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "7220bc21c33e990c14f4a9a319b1d242ebc5b269"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.3.1"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FastPow]]
git-tree-sha1 = "7d961335144dad74de0e1b3a9b60e4d114a78dc2"
uuid = "c0e83750-1142-43a8-81cf-6c956b72b4d1"
version = "0.1.0"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "b5e930ac60b613ef3406da6d4f42c35d8dc51419"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.19"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c2178cfbc0a5a552e16d097fae508f2024de61a3"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.59.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "ef49a187604f865f4708c90e3f431890724e9012"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.59.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "60ed5f1643927479f845b0135bb369b031b541fa"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.14"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[HypertextLiteral]]
git-tree-sha1 = "72053798e1be56026b81d4e2682dbe58922e5ec9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.0"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "34dc30f868e368f8a17b728a1238f3fcda43931a"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.3"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measurements]]
deps = ["Calculus", "LinearAlgebra", "Printf", "RecipesBase", "Requires"]
git-tree-sha1 = "31c8c0569b914111c94dd31149265ed47c238c5b"
uuid = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
version = "2.6.0"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "9d8c00ef7a8d110787ff6f170579846f776133a9"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.4"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "2537ed3c0ed5e03896927187f5f2ee6a4ab342db"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.14"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "457b13497a3ea4deb33d273a6a5ea15c25c0ebd9"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.22.2"

[[PlutoUI]]
deps = ["Base64", "Dates", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "0c3e067931708fa5641247affc1a1aceb53fff06"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.11"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "fca29e68c5062722b5b4435594c3d1ba557072a3"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.7.1"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ad42c30a6204c74d264692e633133dcea0e8b14e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.2"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "1162ce4a6c4b7e31e0e6b14486a6986951c73be9"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.2"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─a87fad48-73c1-4a08-a6f1-aae759b3c6fc
# ╟─172227c2-b27a-40db-91f4-9566c2f6cf52
# ╟─4b484cf6-4888-4f04-b3fd-94862822b0c0
# ╠═a82f22e4-f35b-461a-b481-1dff43722e44
# ╠═8c444ee4-8c77-413a-bbeb-9e5ae2428876
# ╟─a0de01b5-779a-48c0-8d61-12b02a5f527e
# ╠═f87e4036-8f82-41c7-90c1-daa5f677488d
# ╟─532eb3bc-5522-4348-afa5-5336ec6752c7
# ╟─dc5f7484-3dc3-47a7-ad4a-30f97fc14d11
# ╟─ab3ff21b-bf82-4d8c-abd1-c27418956ed8
# ╠═7a1db355-bba9-4322-9fb4-a6d7b7bdd60d
# ╟─cc49ef04-08d8-42bb-9170-9db64e275a51
# ╠═df33b999-4a42-4133-bf59-5a65240790cf
# ╠═0f52365d-34f4-46ed-923e-3ea31c6db0ca
# ╟─d00e56f2-9d3a-4dd3-80eb-3201eff39b96
# ╠═b5c09cd3-6063-4a36-96cd-2d128aa11b82
# ╟─374f239b-6470-40ed-b068-a8ecaace4f09
# ╟─7719b317-e85b-4583-b401-a8614d4b2373
# ╠═f58769a6-a656-42a3-8bc6-c204d4cfd897
# ╟─144119ad-ab88-4165-883a-f2fc2464a838
# ╠═356ac5a4-c94e-42cb-a085-0198b29c7e52
# ╟─43e6b146-ee35-40f1-b540-3da22b9e1b1b
# ╟─dd9e4332-2908-40ef-b461-6b571df56cf4
# ╠═d23b4a92-055e-4ed7-bd46-8a3c59312993
# ╠═e6e29d1e-9a93-49db-a358-6b66f0bc3433
# ╟─017cd6a8-712d-4ec5-b31f-7f1f507764f2
# ╟─b5206dd5-1f46-4437-929b-efd68393b12b
# ╠═eb5dc224-1491-11ec-1cae-d51c93cd292c
# ╟─594ba1d6-2dae-4f20-9546-f52fac17c2f0
# ╟─66c7d960-7e05-4613-84e8-2a40fe40dc3d
# ╟─e717a8d9-ccfb-4f89-b2a2-f244f108b48d
# ╠═3755a4f3-1842-4de2-965e-d294c06c54c7
# ╟─505ef5ab-f131-4ab3-a723-795b5eb5dc0f
# ╟─eab7b195-64d5-4587-8687-48a673ab091b
# ╟─34dc72dc-4864-47c0-b730-183f67e7aea3
# ╠═beeb3335-5c49-47de-a1d3-3eef5f9479f1
# ╟─02d9bf3b-708c-4293-b198-9043b334ff7e
# ╠═0967b90d-ac88-476d-a57a-7c38dfa82204
# ╟─0a5282ee-c88a-4bcc-aca2-477f28e9e04d
# ╠═b2a4a505-47ff-40bb-9a6d-a08d91c53217
# ╟─fcff6973-012a-40fc-a618-f6262266287a
# ╠═985b4ffb-7964-4b50-8c2f-e5f45f352500
# ╟─14867ffd-cde5-43f8-8399-01169ee29b73
# ╟─c4798182-de75-4b59-8be7-f7cf1051364d
# ╠═efc586a2-0946-4dc5-ab3a-3902a811f3ad
# ╟─22fb0386-d4fa-47b9-ac31-decf2731cbc1
# ╟─8e23a3ea-3039-4a5f-b37f-c4710153938e
# ╠═d42f842d-6c2a-40db-b0c4-e936244a9e7c
# ╠═1ad401b5-20b2-489b-b2aa-92f729b1d725
# ╟─2a3e7257-63ad-4761-beda-cec18b91f99c
# ╟─49b1f040-929a-4238-acd9-6554757b592c
# ╠═26d5c6e9-a903-4792-a0e0-dec1a2e86a01
# ╟─2aef27bd-dea6-4a93-9d0f-b9249c9dd2cd
# ╠═0546ee2d-b62d-4c7a-8172-ba87b3c1aea4
# ╟─4a498c18-406f-4437-b378-aa9fdc75b919
# ╟─b6dcb9a3-59e3-4eae-9399-fb072c704f1a
# ╠═e4657169-1bb2-4d4a-ac9d-adc80499d07d
# ╟─5d395353-5681-4780-983e-902fdb89eaf2
# ╠═e9376a4b-3d60-42eb-8681-cd2bcec13fe8
# ╠═c5cdd71f-5ded-482f-9205-c13f01a14d0b
# ╟─a0993a1f-60a6-45b5-815e-676c49a9f049
# ╠═4e7f8db4-b5cc-4a3e-9fa7-e62d8f2a36ac
# ╟─36da3e92-000c-4d4b-9abf-4cd588b3a354
# ╠═70eb2e0a-a5c8-4975-8f6c-589035bea29c
# ╟─b4646a29-3efd-4bd1-bffc-3575559de937
# ╠═d32743c0-fc80-406f-83c5-4528e439589a
# ╟─98478246-5940-4828-a8f1-9c9fa990676d
# ╠═310f247e-3fe8-4621-ae0b-b5ee38d2ee89
# ╟─8267220a-f06e-4761-b310-00f8ba44e4b1
# ╟─ce916139-221a-462e-877f-88212663c05e
# ╠═99b5c818-a825-4939-849e-1cade802f63d
# ╟─8e2903be-4975-4e14-84ed-6e712f47fe47
# ╟─418f31bb-81d5-459b-b402-4fd4e3f4ab27
# ╠═05402cbd-78c6-4234-8680-c351c8c37778
# ╟─4e97f24c-c237-4117-bc57-e4e88c8fb8d2
# ╠═b31da90d-7165-42de-b18d-90584affea03
# ╟─5f37640b-ffd9-4877-a78c-a699b2671919
# ╠═d87c22d1-d595-4d43-ab1c-f28d282a3485
# ╟─1d6eedfd-d013-4557-9cf2-103f8fb7b72a
# ╟─bf0a5303-f5ce-4711-b9ee-a12ce2d8a397
# ╟─c003a61d-a434-4d7b-9214-5b52aa044248
# ╠═e24ce081-e367-4feb-8a79-66b8654a0b3a
# ╟─63eb391f-0238-434a-bc3a-2fa8ed41448e
# ╠═7b9bb0fd-34a5-42e1-bc35-7259447b73d0
# ╟─6a4e0e2e-75c5-4cab-987d-3d6b62f9bb06
# ╠═c91862dd-498a-4712-8e3d-b77e088cd470
# ╠═a08d6e6d-ddc4-40aa-b7c4-93ea03191415
# ╟─a356e2cc-1cb1-457a-986c-998cf1efe008
# ╠═57141f7c-9261-4dc5-98e4-b136a15f86fc
# ╟─055e32d7-073c-40db-a267-750636b9f786
# ╠═aaa97ce4-a5ff-4332-89a2-843cee2e5b6d
# ╟─1067527e-76b7-4331-b3ab-efd72fb99dfc
# ╟─93697e4d-369b-48e9-8b28-a0ff58604d02
# ╟─c4344e64-aa22-4328-a97a-71e44bcd289f
# ╟─827bda6f-87d4-4d36-8d89-f144f4595240
# ╟─1ff4077a-4742-4c5e-a8d6-c4699469a683
# ╠═4a75498d-8f4e-406f-8b01-f6a5f153919f
# ╟─3ae783ce-d06e-4cc2-b8a3-94512e8f1490
# ╠═13e7da81-8581-4f32-9fdb-2599dd36a12c
# ╟─4870b1f3-3134-4ddc-a59d-fa806b456a23
# ╠═fda6171c-9675-4f2e-b226-7ccf100529cd
# ╟─a862f8a3-0131-4644-bc90-246bf3120790
# ╠═d6564250-f646-40de-9463-a956af1a5b1d
# ╟─eee3ac4b-4ddb-4699-b6e6-f0ffcc562c07
# ╠═107aec28-ecb5-4007-95e5-25d0a7f0c465
# ╟─1394e4c6-c371-47c0-8ca8-f0830d63d8ec
# ╠═535716e6-9c1c-4324-a4cd-b1214df3c01d
# ╟─b8edfb4e-6780-4ce7-94c1-4073ff7fa832
# ╠═b8320f78-323c-49a9-a9f9-2748d19ecb35
# ╟─92737d73-676c-4c96-a321-831ecaf37690
# ╠═931a9c5f-8f91-4e88-956b-50c0efc9c58b
# ╟─b5b96082-efde-464f-bcd4-f2e0a84befcd
# ╠═7658a32c-d3da-4ec9-9d96-0d30bb18f08c
# ╟─e61981d5-5448-45e9-81dc-320ac87ba813
# ╠═31e1bb51-c531-4c4a-8634-5caafb7e9e51
# ╠═b0b81da4-6788-45c4-b618-188a02b5e09c
# ╟─47c205c3-ceae-4e12-9ade-753df1608deb
# ╟─4cef9cea-1e84-42b9-bff6-b9a8b3bfe8da
# ╟─826693ff-9a9b-46b1-aeb3-767a5e6f9441
# ╟─d231842d-9b7a-4711-b71b-5d54041ebc1f
# ╟─53cedd26-3742-4c23-a8b8-8a1f2bdfa135
# ╟─889f837d-2e26-4261-b276-5fd91efdda6a
# ╟─670a01e3-82f8-4c7f-8577-852081d91ed7
# ╠═fce1e3e0-cdf7-453f-b913-964c10fa85a6
# ╟─69a92ac6-833c-4605-b3d0-9400e4572886
# ╠═542a9ef5-d9ee-49bd-9d31-61e28b80b5cb
# ╟─8bada25c-b586-42b4-851d-232ccca8a456
# ╠═7600c6dc-769e-4c77-8526-281a1bcec079
# ╠═29dbc47b-3697-4fdf-8f34-890ab4d0cdae
# ╟─0ee7fc18-f41f-4179-a75e-1e1d56b2db29
# ╟─e5b557d7-0952-4409-ae4c-a0c8ce736e03
# ╟─0d0374ed-5150-40e6-b5a4-9a344b6ca47a
# ╠═d2f4d622-8e35-4be3-b421-39b28a748cab
# ╟─f7cf613e-be9d-4f62-a778-cc4375eb99df
# ╟─5be87c6f-5c31-4d14-a8cb-4e63ef39d538
# ╟─0c07edd3-c0a1-4f72-a16a-74badb7a6123
# ╠═4fc5ef4d-e072-41f7-aef9-b42730c8313c
# ╟─19c5cc9d-8304-4e36-a3ea-a1151f28f71d
# ╠═7dcadd85-2986-4e42-aa84-67128a8f666d
# ╟─0b5c6ede-bceb-499a-a9a8-3c6a75ed340a
# ╠═91b5eac1-4799-4a72-ac6a-e2b117b787d5
# ╟─0f86ab3c-29aa-472b-8194-228c736ee940
# ╠═0b8a2292-c0d6-44e4-b560-32d9d579a008
# ╟─d6585cca-78bf-41d1-aea3-01d9831d76cb
# ╠═1b7b7d48-79d2-4317-9045-5b7e7bd073e5
# ╟─3f9dad58-294c-405c-bfc4-67855bb1e825
# ╟─30d2f39e-5df2-4f38-8032-e5f8492ba335
# ╟─6d61b58f-b88f-48f4-8bdd-0bb1a8bc1c82
# ╟─76b8695e-64dc-44bc-8938-ce22c4a9e4d0
# ╟─372637ff-9305-4d45-bf6e-e6531dadbd14
# ╟─3b2c08a6-b27e-49be-a0b0-e5cb3d5546e0
# ╠═f049ab19-7ecf-4c65-bf6d-21352c1fe767
# ╟─7280e368-c68a-48a5-91fe-93c76607c144
# ╠═755fae26-6db9-45a0-a60d-d0e9c063f8aa
# ╟─9a8d8012-ba54-4d9b-8c4c-fe6358508f2a
# ╠═ffbeae5f-8aec-4473-a446-5b73bd911733
# ╟─3738e40f-9596-469d-aa58-a4b28d8a22f8
# ╠═5f1054b8-2337-43c1-a086-26233e95d42b
# ╠═bd719619-bdd4-4c3c-8d66-1df0f210c595
# ╟─f289955b-0239-4b8d-ba08-2edf0a7284c2
# ╠═878ab5f7-28c1-4832-9c58-cb36b360766f
# ╠═a0dcd888-059d-4abe-bb6b-958d2879101c
# ╟─2634feff-7442-4d8f-b8e5-c11113136980
# ╟─b1c5b7e5-cfbf-4d93-bd79-2924d957ae14
# ╠═cd1102ac-1500-4d79-be83-72ac9180c7ce
# ╠═f21604f4-e4f7-4d43-b3d9-32f429df443e
# ╠═10826a95-16f8-416d-b8c1-0ef3347c9b20
# ╟─c46a4f97-78e4-42fd-82b3-4dc6ce99abac
# ╠═7c433791-a653-4836-91e2-084355c01d90
# ╠═410b9da4-7848-4385-bffc-a3d9bd03cf19
# ╟─c08fff28-520e-40af-951c-fe3c324f67e0
# ╟─eb0f9080-2523-4633-be21-3a2281a1629e
# ╠═357c6621-b2b8-4f30-ba41-ffc1ae6f031b
# ╟─e3cc3c77-71ad-4006-8e27-fabaa1ae9cfb
# ╠═58eb5b4b-76ad-4f7a-b86b-0494a857dca1
# ╟─574047fa-6626-4cd0-8317-32118129711e
# ╟─739c9a8a-13a5-4a33-a441-f5bc6cb35e82
# ╟─4e059cb8-6dac-450d-9f46-b3e657d9c3cf
# ╟─5ff9c89a-d999-4af2-8e7e-fb99d4948c36
# ╠═0f2f55f6-060b-475e-bef7-eaa99da4d99f
# ╠═415ad590-247b-4a5d-b21e-7af4d0c17493
# ╟─53c2e16e-b7f5-4df2-96f4-08402b5f8979
# ╠═16cdbc18-e846-4d0a-b7e6-87f07c0c52d9
# ╟─79169f89-fedc-466b-8170-fff99b98e147
# ╠═224336e2-522c-44af-b9a1-307e2ffff0f9
# ╟─06526edf-911a-4ecc-a350-6d932ca56cd5
# ╟─b5e258cd-5542-4a4c-ae0f-91c2fee426db
# ╠═c48210e0-1a04-4f84-a4e2-f6b5d34a603d
# ╟─b0dc1c2b-82b7-488d-8074-1ef9f59a15e5
# ╠═0471b987-f987-4656-b961-285e32d0a5e1
# ╟─97b8b15b-75c7-4321-999d-b067ed2a04f9
# ╟─d9f254dc-ae4a-40b3-b682-f8a501e10a2d
# ╟─591e6a9c-444c-471f-a56b-4dfbc9111989
# ╟─1f265576-824a-4764-a738-685554068079
# ╠═339487cd-8ee8-4d1d-984b-b4c5ff00bae3
# ╟─9cb29b01-7f49-4145-96d8-c8fd971fe1c8
# ╟─ac52a71b-1138-4f1b-99c3-c174d9f09187
# ╟─1e099e1f-6494-419b-8517-5bded3e18aa6
# ╟─10c86547-d4f4-4c3f-8906-ac18ce93f3b6
# ╟─2871aca3-e6b4-4a2d-868a-36562e9a274c
# ╟─2a2e9155-1c77-46fd-8502-8431573f94d0
# ╠═7c792b6b-b6ee-4e30-88d5-d0b8064f2734
# ╟─b557a646-8c3c-4dc7-8788-bf98aec8c5c0
# ╠═febe8c06-b3aa-4db1-a3ea-fdc2a81bdebd
# ╟─260d5753-6cc2-4137-8a2c-8d8a47585ecf
# ╠═a9981931-4cc9-4d16-a6d2-34b4071a84d7
# ╟─b4154fb7-e0b0-4211-8490-8a8fe47cd2da
# ╠═8ac7b1bf-c958-4eb5-8376-f802b372e796
# ╟─b5008faf-fd43-45dd-a5a1-7f51e0b4ede5
# ╠═a756dd18-fac6-4527-944e-c16d8cc4bf95
# ╠═555d1f62-b95b-4377-a8e2-9e442ee7526d
# ╠═f5510c1e-9b9f-49f0-bc7e-0fd8e79a5760
# ╠═7f556f7c-cdb0-4f91-a359-2f933bbc5b68
# ╠═5d9a40b5-4050-47d2-9855-e9b62d56e8df
# ╠═0fc843d2-ac4f-4717-a298-92a476223112
# ╠═fc7f665b-00d9-431b-a97e-d2ff7253221a
# ╠═aadf6e48-0cbf-4973-86a1-173b6648d1df
# ╠═452e1ea7-98be-4910-ba6b-c0881fb251b2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
