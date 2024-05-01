import numpy as np
from scipy.stats import nakagami
import matplotlib.pyplot as plt
 
N = 100000  # número de repetições na simulação de Monte Carlo
PdB = 10  # potência de transmissão em dB
P = 10**(PdB/10)  # potência de transmissão linear
N0 = 1  # densidade espectral de potência do ruído
R_range = np.arange(0.1, 5.1, 0.1)  # vetor de taxas de transmissão
 
# Simulação para obter a Outage Probability do link direto
 
# função de cálculo da Informação Mútua
def imutua_direto(hij, P, N0):
    return np.log2(1 + ((np.abs(hij)**2) * P) / N0)
 
co = np.zeros(len(R_range))  # número de falhas na comunicação
probs_direto = np.zeros(len(R_range))  # vetor probabilidade em função de R
 
# modelamento do canal com link direto
hij = 1 * nakagami.rvs(0.5, size=N)
 
# Simulando e obtendo as probabilidades por Monte Carlo
for j in range(len(R_range)):
    for i in range(N):
        if imutua_direto(hij[i], P, N0) < R_range[j]:
            co[j] += 1
    probs_direto[j] = co[j] / N
 
# Simulação para obter a Outage Probability utilizando a técnica VHD
co_vhd_sd = np.zeros(len(R_range))  # nº de falhas no link direto
co_vhd_sr = np.zeros(len(R_range))  # nº de falhas no link source-relay
co_vhd_rd = np.zeros(len(R_range))  # nº de falhas no link relay-destination
 
probs_vhd_sd = np.zeros(len(R_range))  # Outage Probability em função de R no link direto
probs_vhd_sr = np.zeros(len(R_range))  # Outage Probability em função de R no link source-relay
probs_vhd_rd = np.zeros(len(R_range))  # Outage Probability em função de R no link relay-destination
probs_vhd = np.zeros(len(R_range))  # Outage Probability em função de R com VHD
 
# Média para modelamento do canal
fator = np.power(0.5, -4)
 
# modelamento dos canais direto, source-relay e relay-destination com VHD
hij_vhd_sd = 1 * nakagami.rvs(0.5, size=N)
hij_vhd_sr = np.sqrt(fator) * nakagami.rvs(1, size=N)
hij_vhd_rd = np.sqrt(fator) * nakagami.rvs(1, size=N)
 
 
# função de cálculo da Informação Mútua dos links source-k
def imutua_vhd_sk(hsk, P, N0):
    return 0.5 * np.log2(1 + ((np.abs(hsk)**2) * P) / N0)
 
# função de cálculo da Informação Mútua do link relay-destination
def imutua_vhd_rd(hsd, hrd, P, N0):
    return 0.5 * np.log2(1 + (((np.abs(hsd)**2) * P) + (np.abs(hrd**2)) * P) / N0)
 
# Simulando e obtendo as probabilidades por Monte Carlo
# P/ m = 0 -> link direto
# P/ m = 1 -> link source-relay
# P/ m = 2 -> link relay-destination
for m in range(3):
    for j in range(len(R_range)):
        for i in range(N):
            if m == 0:
                if imutua_vhd_sk(hij_vhd_sd[i], P, N0) < R_range[j]:
                    co_vhd_sd[j] += 1
            elif m == 1:
                if imutua_vhd_sk(hij_vhd_sr[i], P, N0) < R_range[j]:
                    co_vhd_sr[j] += 1
            else:
                if imutua_vhd_rd(hij_vhd_sd[i], hij_vhd_rd[i], P, N0) < R_range[j]:
                    co_vhd_rd[j] += 1
            probs_vhd_sr[j] = co_vhd_sr[j] / N
            probs_vhd_sd[j] = co_vhd_sd[j] / N
            probs_vhd_rd[j] = co_vhd_rd[j] / N
 
# cálculo da Outage Probability com VHD
probs_vhd = probs_vhd_sd * probs_vhd_sr + (1 - probs_vhd_sr) * probs_vhd_rd
 
 
# Simulação para obter a Outage Probability utilizando a técnica VJD
co_vjd_sd = np.zeros(len(R_range))  # nº de falhas no link direto
co_vjd_sr = np.zeros(len(R_range))  # nº de falhas no link source-relay
co_vjd_rd = np.zeros(len(R_range))  # nº de falhas no link relay-destination
 
probs_vjd_sd = np.zeros(len(R_range))  # Outage Probability em função de R no link direto
probs_vjd_sr = np.zeros(len(R_range))  # Outage Probability em função de R no link source-relay
probs_vjd_rd = np.zeros(len(R_range))  # Outage Probability em função de R no link relay-destination
probs_vjd = np.zeros(len(R_range))  # Outage Probability em função de R com VJD
 
# Média para modelamento do canal
fator = np.power(0.5, -4)
 
# modelamento dos canais direto, source-relay e relay-destination com VJD
# Usando o mesmo fator de atenuação VHD para modelar os canais e VJD
hij_vjd_sd = 1 * nakagami.rvs(0.5, size=N)
hij_vjd_sr = np.sqrt(fator) * nakagami.rvs(1, size=N)
hij_vjd_rd = np.sqrt(fator) * nakagami.rvs(1, size=N)
 
hij_vjd_rr = np.sqrt(10**-4) * nakagami.rvs(1, size=N)  # interferência no relay devido ao modo full-duplex
 
# função de cálculo da Informação Mútua do link direto source-destination
def imutua_vjd_sd(hsd, P, N0):
    return np.log2(1 + ((np.abs(hsd) ** 2) * P) / N0)
 
# função de cálculo da Informação Mútua do link source-relay
def imutua_vjd_sr(hsr, hrr, P, N0):
    return np.log2(1 + ((np.abs(hsr) ** 2) * P) / (N0 + P*(np.abs(hrr) ** 2) ))
 
# função de cálculo da Informação Mútua do link relay-destination
def imutua_vjd_rd(hsd, hrd, P, N0):
    return np.log2(1 + (((np.abs(hsd) * 2) * P) + (np.abs(hrd * 2)) * P) / N0)
 
# Simulando e obtendo as probabilidades por Monte Carlo
# P/ m = 0 --> link direto
# P/ m = 1 --> link source-relay
# P/ m = 2 --> link relay-destination
for m in range(3):
    for j in range(len(R_range)):
        for i in range(N):
            if m == 0:
                if imutua_vjd_sd(hij_vjd_sd[i], P, N0) < R_range[j]:
                    co_vjd_sd[j] += 1
            elif m == 1:
                if imutua_vjd_sr(hij_vjd_sr[i], hij_vjd_rr[i], P, N0) < R_range[j]:
                    co_vjd_sr[j] += 1
            else:
                if imutua_vjd_rd(hij_vjd_sd[i], hij_vjd_rd[i], P, N0) < R_range[j]:
                    co_vjd_rd[j] += 1
            probs_vjd_sr[j] = co_vjd_sr[j] / N
            probs_vjd_sd[j] = co_vjd_sd[j] / N
            probs_vjd_rd[j] = co_vjd_rd[j] / N
 
# cálculo da Outage Probability com VJD
probs_vjd = probs_vjd_sd * probs_vjd_sr + (1 - probs_vjd_sr) * probs_vjd_rd
 
 
# Simulação para obter a Outage Probability utilizando a técnica VDH
co_vdh_sr = np.zeros(len(R_range))  # nº de falhas no link source-relay
co_vdh_rd = np.zeros(len(R_range))  # nº de falhas no link relay-destination
 
probs_vdh_sr = np.zeros(len(R_range))  # Outage Probability em função de R no link source-relay
probs_vdh_rd = np.zeros(len(R_range))  # Outage Probability em função de R no link relay-destination
probs_vdh = np.zeros(len(R_range))  # Outage Probability em função de R com VDH
 
# Média para modelamento do canal
fator = np.power(0.5, -4)
 
# modelamento dos canais direto,source-relay e relay-destination com VDH
hij_vdh_sd = 1 * nakagami.rvs(0.5, size=N)
hij_vdh_sr = np.sqrt(fator) * nakagami.rvs(1, size=N)
hij_vdh_rd = np.sqrt(fator) * nakagami.rvs(1, size=N)
 
hij_vdh_rr = np.sqrt(10**-4) * nakagami.rvs(1, size=N)  # interferência no relay devido ao modo full-duplex
 
# função de cálculo da Informação Mútua do link source-relay
def imutua_vdh_sr(hsr, hrr, P, N0):
    return np.log2(1 + ((np.abs(hsr) ** 2) * P) / (N0 + P*(np.abs(hrr) ** 2) ))
 
# função de cálculo da Informação Mútua do link relay-destination
def imutua_vdh_rd(hsd, hrd, P, N0):
    return np.log2(1 + (((np.abs(hrd) ** 2) * P) / (N0+(np.abs(hsd ** 2)) * P)))
 
# Simulando e obtendo as probabilidades por Monte Carlo
# P/ m = 0 --> link source-relay
# P/ m = 1 --> link relay-destination
for m in range(2):
    for j in range(len(R_range)):
        for i in range(N):
            if m == 0:
                if imutua_vdh_sr(hij_vdh_sr[i], hij_vdh_rr[i], P, N0) < R_range[j]:
                    co_vdh_sr[j] += 1
            else:
                if imutua_vdh_rd(hij_vdh_sd[i], hij_vdh_rd[i], P, N0) < R_range[j]:
                    co_vdh_rd[j] += 1
            probs_vdh_sr[j] = co_vdh_sr[j] / N
            probs_vdh_rd[j] = co_vdh_rd[j] / N
 
# cálculo da Outage Probability com VDH
probs_vdh = probs_vdh_sr + probs_vdh_rd - probs_vdh_sr * probs_vdh_rd
 
 
# plotando as curvas de Outage Probability vs R
plt.semilogy(R_range, probs_direto, 'o-', label='Direto', color='purple')  # Curva do link direto em roxo
plt.semilogy(R_range, probs_vhd, 'x-', label='VHD', color='green')       # Curva do VHD em verde
plt.semilogy(R_range, probs_vjd, '+-', label='VJD', color='blue')         # Curva do VJD em azul
plt.semilogy(R_range, probs_vdh, '*-', label='VDH', color='orange')      # Curva do VDH em laranja
plt.xlabel('R')
plt.ylabel('Outage Probability')
plt.title('Outage probability for the different schemes as a function of transmission rate R')
plt.ylim(1e-3, 1)  # Define o limite do eixo Y de 10^-3 a 1
plt.xlim(1, 5)  # Define o limite do eixo X de 1 a 6
plt.legend()
plt.grid(axis='both')
plt.show()
