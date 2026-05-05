import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def obtener_datos(ticker: str, inicio: str, fin: str) -> pd.DataFrame:
    print(f"\n📥 Descargando datos de {ticker} desde {inicio} hasta {fin}...")

    df = yf.download(ticker, start=inicio, end=fin, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No se encontraron datos para el ticker '{ticker}'.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"✅ {len(df)} registros descargados correctamente.")
    return df


def calcular_medias_moviles(df: pd.DataFrame,
                            periodo_corto: int = 10,
                            periodo_largo: int = 50) -> pd.DataFrame:
    df['SMA_Corta'] = df['Close'].rolling(window=periodo_corto).mean()
    df['SMA_Larga'] = df['Close'].rolling(window=periodo_largo).mean()

    print(f"\n📊 Medias móviles calculadas:")
    print(f"   → SMA Corta : {periodo_corto} períodos")
    print(f"   → SMA Larga : {periodo_largo} períodos")

    return df


def generar_senales(df: pd.DataFrame) -> pd.DataFrame:
    df['Señal'] = 0

    df_valido = df.dropna(subset=['SMA_Corta', 'SMA_Larga'])

    for i in range(1, len(df_valido)):
        idx_actual   = df_valido.index[i]
        idx_anterior = df_valido.index[i - 1]

        sma_corta_ant = df_valido.loc[idx_anterior, 'SMA_Corta']
        sma_larga_ant = df_valido.loc[idx_anterior, 'SMA_Larga']

        sma_corta_act = df_valido.loc[idx_actual, 'SMA_Corta']
        sma_larga_act = df_valido.loc[idx_actual, 'SMA_Larga']

        if sma_corta_ant <= sma_larga_ant and sma_corta_act > sma_larga_act:
            df.loc[idx_actual, 'Señal'] = 1

        elif sma_corta_ant >= sma_larga_ant and sma_corta_act < sma_larga_act:
            df.loc[idx_actual, 'Señal'] = -1

    return df


def simular_trading(df: pd.DataFrame, capital_inicial: float = 1000.0) -> dict:
    capital       = capital_inicial
    acciones      = 0.0
    en_posicion   = False
    historial     = []
    num_operaciones = 0

    print(f"\n💰 Capital inicial: ${capital_inicial:,.2f}")
    print("=" * 60)
    print("📋 HISTORIAL DE OPERACIONES:")
    print("-" * 60)

    for fecha, fila in df.iterrows():
        precio  = float(fila['Close'])
        senal   = int(fila['Señal'])

        if senal == 1 and not en_posicion:
            acciones   = capital / precio
            capital    = 0.0
            en_posicion = True
            num_operaciones += 1

            operacion = {
                'Fecha' : fecha.date(),
                'Tipo'  : 'COMPRA',
                'Precio': precio,
                'Acciones': round(acciones, 4),
                'Capital': 0.0
            }
            historial.append(operacion)
            print(f"🟢 COMPRA  | {fecha.date()} | Precio: ${precio:>8.2f} | "
                  f"Acciones: {acciones:.4f}")

        elif senal == -1 and en_posicion:
            capital    = acciones * precio
            acciones   = 0.0
            en_posicion = False
            num_operaciones += 1

            operacion = {
                'Fecha' : fecha.date(),
                'Tipo'  : 'VENTA',
                'Precio': precio,
                'Acciones': 0,
                'Capital': round(capital, 2)
            }
            historial.append(operacion)
            print(f"🔴 VENTA   | {fecha.date()} | Precio: ${precio:>8.2f} | "
                  f"Capital: ${capital:,.2f}")

    ultimo_precio = float(df['Close'].iloc[-1])
    if en_posicion:
        capital = acciones * ultimo_precio
        print(f"\n⚠️  Posición abierta al final — valorada a ${capital:,.2f}")

    ganancia = capital - capital_inicial
    rendimiento = (ganancia / capital_inicial) * 100

    resultados = {
        'capital_inicial'  : capital_inicial,
        'capital_final'    : round(capital, 2),
        'ganancia'         : round(ganancia, 2),
        'rendimiento_pct'  : round(rendimiento, 2),
        'num_operaciones'  : num_operaciones,
        'historial'        : historial
    }

    return resultados


def mostrar_resultados(resultados: dict) -> None:
    ganancia = resultados['ganancia']
    emoji    = "📈" if ganancia >= 0 else "📉"
    signo    = "+" if ganancia >= 0 else ""

    print("\n" + "=" * 60)
    print("           📊  RESUMEN DE LA SIMULACIÓN  📊")
    print("=" * 60)
    print(f"  💵 Capital inicial   : ${resultados['capital_inicial']:>10,.2f}")
    print(f"  💰 Capital final     : ${resultados['capital_final']:>10,.2f}")
    print(f"  {emoji}  Ganancia / Pérdida : {signo}${abs(ganancia):>9,.2f}")
    print(f"  📉 Rendimiento       : {signo}{resultados['rendimiento_pct']}%")
    print(f"  🔁 Operaciones totales: {resultados['num_operaciones']:>9}")
    print("=" * 60)


def graficar_resultados(df: pd.DataFrame, ticker: str,
                        resultados: dict) -> None:
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df.index, df['Close'],
            label='Precio de Cierre',
            color='#a0c4ff',
            linewidth=1.2,
            alpha=0.8,
            zorder=2)

    ax.plot(df.index, df['SMA_Corta'],
            label='SMA Corta (10)',
            color='#ffd166',
            linewidth=1.8,
            linestyle='--',
            zorder=3)

    ax.plot(df.index, df['SMA_Larga'],
            label='SMA Larga (50)',
            color='#ef476f',
            linewidth=1.8,
            linestyle='--',
            zorder=3)

    compras = df[df['Señal'] == 1]
    ax.scatter(compras.index, compras['Close'],
               marker='^',
               color='#06d6a0',
               s=120,
               label='Compra',
               zorder=5)

    ventas = df[df['Señal'] == -1]
    ax.scatter(ventas.index, ventas['Close'],
               marker='v',
               color='#ff6b6b',
               s=120,
               label='Venta',
               zorder=5)

    for fecha, fila in compras.iterrows():
        ax.annotate(f"${fila['Close']:.0f}",
                    xy=(fecha, fila['Close']),
                    xytext=(0, 12),
                    textcoords='offset points',
                    fontsize=7,
                    color='#06d6a0',
                    ha='center')

    for fecha, fila in ventas.iterrows():
        ax.annotate(f"${fila['Close']:.0f}",
                    xy=(fecha, fila['Close']),
                    xytext=(0, -16),
                    textcoords='offset points',
                    fontsize=7,
                    color='#ff6b6b',
                    ha='center')

    ax.set_title(
        f"Bot de Trading — {ticker} | "
        f"Capital: ${resultados['capital_inicial']:,.0f} → "
        f"${resultados['capital_final']:,.2f} "
        f"({'▲' if resultados['ganancia'] >= 0 else '▼'}"
        f"{resultados['rendimiento_pct']}%)",
        fontsize=13, fontweight='bold', pad=15
    )
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("Precio (USD)", fontsize=11)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    ax.legend(loc='upper left', fontsize=10, framealpha=0.4)
    ax.grid(color='#333333', linestyle='--', linewidth=0.5, alpha=0.6)

    plt.tight_layout()

    ruta = "trading_resultado.png"
    plt.savefig(ruta, dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e')
    print(f"\n🖼️  Gráfica guardada como '{ruta}'")
    plt.show()


def main():
    TICKER          = "AAPL"
    FECHA_INICIO    = "2020-01-01"
    FECHA_FIN       = "2024-12-31"
    CAPITAL_INICIAL = 1_000.0
    SMA_CORTA       = 10
    SMA_LARGA       = 50

    print("=" * 60)
    print("    BOT DE TRADING ALGORÍTMICO — MODO SIMULACIÓN")
    print("     Solo datos históricos. Sin dinero real.")
    print("=" * 60)

    df = obtener_datos(TICKER, FECHA_INICIO, FECHA_FIN)
    df = calcular_medias_moviles(df, SMA_CORTA, SMA_LARGA)
    df = generar_senales(df)
    resultados = simular_trading(df, CAPITAL_INICIAL)
    mostrar_resultados(resultados)
    graficar_resultados(df, TICKER, resultados)


if __name__ == "__main__":
    main()