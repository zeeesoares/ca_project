#!/usr/bin/env python3

import re
import matplotlib.pyplot as plt


def parse_migrater_v2(file_path):
    timestamps = []
    rates = []

    ts_pattern = re.compile(r"timestamp=([\d\.]+)")
    rate_pattern = re.compile(r"rate=(\d+) B/s")
    migrating_pattern = re.compile(r"migrating=(\w+)")

    start_ts = None

    with open(file_path, 'r') as f:
        for line in f:
            ts_match = ts_pattern.search(line)
            migrating_match = migrating_pattern.search(line)
            rate_match = rate_pattern.search(line)

            if ts_match and migrating_match:
                # Extrair tempo relativo em segundos
                current_ts = float(ts_match.group(1))
                if start_ts is None:
                    start_ts = current_ts

                relative_time = current_ts - start_ts
                is_migrating = migrating_match.group(1) == "True"

                # Extrair Rate
                if is_migrating and rate_match:
                    rate_mib = int(rate_match.group(1)) / (1024 * 1024)
                else:
                    rate_mib = 0.0

                timestamps.append(relative_time)
                rates.append(rate_mib)

    return timestamps, rates

def plot_performance(x, y):
    plt.figure(figsize=(12, 6))

    # Usamos fill_between para destacar a área de migração
    plt.plot(x, y, color='#1f77b4', label='Taxa de Migração (MiB/s)', linewidth=1.5)
    plt.fill_between(x, y, color='#1f77b4', alpha=0.3)

    plt.title('Desempenho da Migração de Checkpoint (Orquestrado)')
    plt.xlabel('Tempo de Treino (segundos desde o início)')
    plt.ylabel('Throughput Efetivo (MiB/s)')

    # Ajustar limites para melhor visualização
    if y:
        plt.ylim(0, max(y) * 1.2)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    output_file = "v2_migrater_plot.png"
    plt.savefig(output_file)
    print(f"Gráfico gerado: {output_file}")
    plt.show()


if __name__ == "__main__":
    log_file = "migrater_output.log"
    x, y = parse_migrater_v2(log_file)

    if x:
        print(f"Total de tempo monitorizado: {x[-1]:.2f} segundos")
        plot_performance(x, y)
    else:
        print("Nenhum dado encontrado com o formato especificado.")
