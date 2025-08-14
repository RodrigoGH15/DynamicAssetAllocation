"""
Proyecto Final: Diploma en Ciencia de Datos para las Finanzas
Sistema Adaptativo de Asignación y Cobertura de Activos basado en
Regímenes de Mercado
"""

from datetime import date
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from dotenv import load_dotenv
from keras import Input
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

load_dotenv(override=True)
TICKERS = {
    "riesgo": "SPY",  # Core Riesgo: S&P 500 ETF
    "seguro": "TLT",  # Core Seguro: Bonos del Tesoro
    "crecimiento": "QQQ",  # Táctico Crecimiento: NASDAQ 100 ETF
    "panico": "^VIX",  # Indicador de Pánico: Índice de Volatilidad
}
# Parámetros del modelo de regímenes
N_REGIMES = 3  # Número de "climas" de mercado a identificar
# Parámetros de la estrategia y backtesting
LOOKAHEAD_DAYS = 21
GROWTH_THRESHOLD = 0.02
# QQQ debe superar a SPY en un 2% para ser considerado éxito
HEDGE_THRESHOLD = -0.05  # Caída de SPY del 5% para justificar la cobertura
CONFIDENCE_THRESHOLD = 0.65
# Probabilidad mínima de la ANN para tomar una acción táctica


def fetch_data(
    tickers: dict[str, str], from_date: date, to_date: date
) -> pd.DataFrame:
    """Descarga datos históricos de precios desde yfinance."""
    try:
        prices = pd.read_excel("prices_data.xlsx")
        print("Se leyeron los datos desde archivo local...")
        if "Date" in prices.columns:
            prices.set_index("Date", inplace=True)
        return prices
    except FileNotFoundError:
        print("Archivo local no encontrado. Descargando desde yfinance...")

    data = yf.download(
        list(tickers.values()), start=from_date, end=to_date, auto_adjust=False
    )
    if data is None:
        raise ValueError(
            "No se pudieron descargar los datos de Yahoo Finance. Verifique "
            "los tickers o la conexión a internet."
        )
    prices = cast(pd.DataFrame, data["Adj Close"])
    # El VIX no es un precio, es un índice, así que se maneja por separado
    vix_ticker = tickers["panico"]
    vix_data = data["Close"][vix_ticker]
    prices[vix_ticker] = vix_data
    prices = prices.ffill().dropna()
    print("Datos descargados y limpios.")
    return prices


def calculate_features(prices: pd.DataFrame):
    """Calcula las features que describen el estado del mercado."""
    print("Calculando features para detección de regímenes...")
    spy_prices = prices[TICKERS["riesgo"]]
    tlt_prices = prices[TICKERS["seguro"]]
    vix_levels = prices[TICKERS["panico"]]

    features = pd.DataFrame(index=prices.index)

    # Usamos log-retornos en vez de los retornos simples
    features["volatility"] = cast(
        pd.Series, np.log(spy_prices / spy_prices.shift(1))
    ).rolling(window=30).std() * np.sqrt(252)  # vol se anualiza por la raíz

    features["trend"] = np.log(
        spy_prices / spy_prices.rolling(window=200).mean()
    )

    features["correlation"] = (
        cast(pd.Series, np.log(spy_prices / spy_prices.shift(1)))
        .rolling(window=60)
        .corr(cast(pd.Series, np.log(tlt_prices / tlt_prices.shift(1))))
    )

    features["vix_level"] = vix_levels

    print("Features calculadas.")
    return features.dropna()


def detect_regimes(features: pd.DataFrame, n_regimes: int):
    """Aplica K-Means para identificar regímenes de mercado."""
    print(f"Detectando {n_regimes} regímenes de mercado...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init="auto")
    features["regime"] = kmeans.fit_predict(scaled_features)

    # Análisis para interpretar los regímenes
    regime_analysis = features.groupby("regime").mean()
    print("\n--- Análisis de Regímenes (Características Promedio) ---")
    print(regime_analysis)
    print("-----------------------------------------------------\n")

    return features


def prepare_ann_data(features: pd.DataFrame, prices: pd.DataFrame):
    """Prepara los datos para entrenar las redes neuronales."""
    print("Preparando datos para las redes neuronales...")
    # Crear variables objetivo (y)
    spy_rets = (
        prices[TICKERS["riesgo"]]
        .pct_change(LOOKAHEAD_DAYS)
        .shift(-LOOKAHEAD_DAYS)
    )
    qqq_rets = (
        prices[TICKERS["crecimiento"]]
        .pct_change(LOOKAHEAD_DAYS)
        .shift(-LOOKAHEAD_DAYS)
    )

    # y_growth: 1 si QQQ superó a SPY por el umbral
    y_growth = (qqq_rets > spy_rets + GROWTH_THRESHOLD).astype(int)

    # y_hedge: 1 si SPY cayó por debajo del umbral
    y_hedge = (spy_rets < HEDGE_THRESHOLD).astype(int)

    data_for_ann = features.copy()
    data_for_ann["y_growth"] = y_growth
    data_for_ann["y_hedge"] = y_hedge
    data_for_ann = data_for_ann.dropna()

    X = data_for_ann[["volatility", "trend", "correlation", "vix_level"]]

    # Normalizar X
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    return (
        X_scaled,
        data_for_ann["y_growth"],
        data_for_ann["y_hedge"],
        data_for_ann.index,
    )


def build_and_train_ann(X, y, name=""):
    """Construye, compila y entrena una red neuronal para clasificación."""
    print(f"Entrenando ANN para '{name}'...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Sequential(
        [
            Input(shape=(X_train.shape[1],)),
            Dense(16, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(8, activation="relu", kernel_regularizer=l2(0.01)),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    print(f"ANN '{name}' entrenada.")
    return model


def run_backtest(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    growth_model: Sequential,
    hedge_model: Sequential,
) -> pd.DataFrame:
    """Ejecuta la simulación de la estrategia adaptativa vs benchmarks."""
    print("Ejecutando backtest de la estrategia...")

    # Preparar datos para el backtest
    backtest_data = features.copy()
    backtest_data = backtest_data.reindex(prices.index).ffill()

    # Predecir probabilidades para todo el set
    X_full_scaled = StandardScaler().fit_transform(
        backtest_data[["volatility", "trend", "correlation", "vix_level"]]
    )
    backtest_data["prob_growth"] = growth_model.predict(X_full_scaled)
    backtest_data["prob_hedge"] = hedge_model.predict(X_full_scaled)

    # Calcular retornos diarios para el backtest
    daily_returns = prices.pct_change().dropna()
    backtest_data = backtest_data.reindex(daily_returns.index).ffill().dropna()
    daily_returns = daily_returns.reindex(backtest_data.index)

    # Inicializar portafolios
    portfolio = pd.DataFrame(index=backtest_data.index)
    portfolio["adaptive_return"] = 0.0

    # Estrategias benchmark
    portfolio["spy_buy_hold"] = daily_returns[TICKERS["riesgo"]]
    portfolio["60_40"] = (
        0.6 * daily_returns[TICKERS["riesgo"]]
        + 0.4 * daily_returns[TICKERS["seguro"]]
    )

    # Bucle de simulación
    for i in range(len(backtest_data)):
        prob_g = backtest_data["prob_growth"].iloc[i]
        prob_h = backtest_data["prob_hedge"].iloc[i]

        # Aplicar reglas de negocio
        if prob_g > CONFIDENCE_THRESHOLD:
            # Tilt a Crecimiento
            ret = (
                0.6 * daily_returns[TICKERS["crecimiento"]].iloc[i]
                + 0.4 * daily_returns[TICKERS["seguro"]].iloc[i]
            )
        elif prob_h > CONFIDENCE_THRESHOLD:
            vix_ret = daily_returns[TICKERS["panico"]].iloc[i] * 0.5
            ret = (
                0.5 * daily_returns[TICKERS["riesgo"]].iloc[i]
                + 0.4 * daily_returns[TICKERS["seguro"]].iloc[i]
                + 0.1 * vix_ret
            )
        else:
            # Portafolio Base 60/40
            ret = portfolio["60_40"].iloc[i]

        # Corrected line using .loc for proper assignment
        portfolio.loc[portfolio.index[i], "adaptive_return"] = ret

    print("Backtest completado.")
    return portfolio


def calculate_performance_metrics(returns_series: pd.Series) -> tuple:
    """Calcula métricas de rendimiento para una serie de log-retornos."""
    days = len(returns_series)

    if days == 0:
        return 0.0, 0.0, 0.0, 0.0

    annualized_return = np.exp(returns_series.mean() * 252) - 1

    annualized_volatility = returns_series.std() * np.sqrt(252)

    sharpe_ratio = annualized_return / annualized_volatility

    cumulative_returns_ts = cast(pd.Series, np.exp(returns_series.cumsum()))
    peak = cumulative_returns_ts.cummax()
    drawdown = (cumulative_returns_ts - peak) / peak
    max_drawdown = drawdown.min()

    return annualized_return, annualized_volatility, sharpe_ratio, max_drawdown


def plot_results(
    backtest_portfolio: pd.DataFrame,
    data_with_regimes,
    from_date: date,
    to_date: date,
):
    """Genera los gráficos de resultados."""
    print("Generando gráficos...")

    cumulative_returns = backtest_portfolio.cumsum()

    sns.set_style("whitegrid")

    fig, ax1 = plt.subplots(figsize=(14, 7))

    cumulative_returns["adaptive_return"].plot(
        ax=ax1, color="crimson", lw=2, label="Estrategia Adaptativa ANN"
    )
    cumulative_returns["spy_buy_hold"].plot(
        ax=ax1, color="navy", lw=1.5, ls="--", label="Benchmark: S&P 500 (SPY)"
    )
    cumulative_returns["60_40"].plot(
        ax=ax1,
        color="darkgray",
        lw=1.5,
        ls=":",
        label="Benchmark: 60/40 SPY/TLT",
    )

    ax1.set_title(
        "Rendimiento Acumulado de la Estrategia Adaptativa vs. Benchmarks",
        fontsize=16,
    )
    ax1.set_ylabel("Rendimiento Acumulado")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    plt.tight_layout()
    plt.show()

    # Gráfico 2: Regímenes de Mercado a lo largo del tiempo
    _, ax2 = plt.subplots(figsize=(14, 7))
    spy_prices = fetch_data(TICKERS, from_date=from_date, to_date=to_date)[
        TICKERS["riesgo"]
    ]
    spy_prices = spy_prices.reindex(data_with_regimes.index)

    ax2.plot(
        spy_prices.index,
        spy_prices.values,  # type: ignore
        color="black",
        label="S&P 500 (SPY)",
        lw=0.5,
    )

    regime_colors = {0: "green", 1: "red", 2: "orange", 3: "blue"}
    for i in range(len(data_with_regimes) - 1):
        ax2.axvspan(
            data_with_regimes.index[i],
            data_with_regimes.index[i + 1],
            facecolor=regime_colors.get(
                data_with_regimes["regime"].iloc[i], "gray"
            ),
            alpha=0.3,
        )

    ax2.set_title(
        "Regímenes de Mercado sobre el Precio del S&P 500", fontsize=16
    )
    ax2.set_ylabel("Precio SPY")
    ax2.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


# --- FUNCIÓN PRINCIPAL ---
def main():
    from_date = date(2018, 1, 1)
    ref_date = date(2025, 6, 30)

    prices_data = fetch_data(TICKERS, from_date, ref_date)
    file_path = Path("prices_data.xlsx")
    if not file_path.is_file():
        pd.DataFrame(prices_data).to_excel(file_path)
        print(f"File '{file_path}' created successfully.")
    else:
        print(f"File '{file_path}' already exists.")
    features_data = calculate_features(prices_data)

    data_with_regimes = detect_regimes(features_data, N_REGIMES)

    X, y_growth, y_hedge, valid_dates = prepare_ann_data(
        data_with_regimes, prices_data
    )
    growth_ann_model = build_and_train_ann(X, y_growth, "Crecimiento")
    hedge_ann_model = build_and_train_ann(X, y_hedge, "Cobertura")

    # Paso 5
    backtest_portfolio = run_backtest(
        data_with_regimes.loc[valid_dates],
        prices_data,
        growth_ann_model,
        hedge_ann_model,
    )

    # Análisis de Métricas de Rendimiento
    metrics = {}
    for col in backtest_portfolio.columns:
        metrics[col] = calculate_performance_metrics(backtest_portfolio[col])
    print("hola")
    metrics_df = pd.DataFrame.from_dict(
        metrics,
        orient="index",
        columns=[
            "Retorno Anualizado",
            "Volatilidad Anualizada",
            "Sharpe Ratio",
            "Max Drawdown",
        ],
    )
    # print("---------------------------------------------------\n")
    print("\n--- Métricas de Rendimiento de las Estrategias ---")
    print(metrics_df)
    print("---------------------------------------------------\n")

    # Paso 6
    plot_results(
        backtest_portfolio,
        data_with_regimes,
        from_date=from_date,
        to_date=ref_date,
    )

    """
    Análisis de la Estrategia Adaptativa vs. SPY y 60/40

    Resumen:
    La estrategia adaptativa (ANN) muestra un rendimiento superior en
    períodos de alta volatilidad, como el COVID-19, donde logra amortiguar
    fuertes caídas del mercado. Sin embargo, desde marzo de 2021 en adelante,
    el SPY toma la delantera y mantiene un retorno superior de manera
    sistemática, aunque la estrategia adaptativa sigue demostrando su capacidad
    para proteger contra caídas abruptas.

    Observaciones principales:
    1. La estrategia adaptativa comienza superando al SPY, con una ventaja
    significativa durante el COVID-19 debido a su capacidad de adaptación
    a la volatilidad del mercado.
    2. En el análisis comparativo con un portafolio 60/40, la estrategia
    adaptativa logra un rendimiento superior a partir del final del COVID.
    3. Los regímenes de mercado se clasifican en tres colores:
    - Rojo: Mercados de alta incertidumbre y caída (2018, 2020 COVID, 2022
    subida de tasas, inflación y eventual caída de SVB, etc.)
    - Amarillo: Períodos de estabilidad o calma relativa (2019, 2020-2 a
    2021-1)
    - Verde: Momentos de tendencia alcista, donde la estrategia adaptativa
    sigue un comportamiento más estable y con riesgo controlado
    4. Las caídas más grandes se amortiguan en los regímenes rojos, lo que
    demuestra la capacidad del modelo para adaptarse a condiciones extremas
    de mercado.
    5. A pesar de no alcanzar el imsmo retorno anualizado que el SPY, la
    estrategia adapativa logra igualar su Sharpe Ratio, lo que sugiere
    una eficiencia riesgo retorno similar con una manor volatilidad y
    un menor drawdown.

    Conclusiones:
    - La estrategia adaptativa se desempeña bien en momentos de alta
    volatilidad, destacándose en períodos como el COVID y otras crisis de
    mercado.
    - Aunque no siempre lidera el rendimiento en comparación con el SPY, el
    modelo demuestra su capacidad para proteger el portafolio durante los
    descensos abruptos, lo que podría ser útil para un inversor más
    conservador.
    - En base a lo anterior, la estrategia adaptativa puede ser más
    adecuada para entornos de mercado altamente inciertos.
    - La estrategia también logra un equilibrio interesante: reduce el riesgo
    en comparación con una inversión 100% en acciones y mejora la eficiencia
    frente a una cartera 60/40. Esto la hace atractiva para perfiles de
    inversión que buscan optimizar la relación riesgo retorno sin renunciar
    completamente al crecimiento.
    - Para mejorar la estrategia, sería interesante ajustar la cantidad de
    regímenes utilizados en el modelo, así como probar diferentes parámetros
    de regularización para optimizar su capacidad de adaptación.
    """


if __name__ == "__main__":
    from rich import print

    main()
