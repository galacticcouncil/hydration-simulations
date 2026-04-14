from datetime import date, timedelta

from hydradx.apps.stableswap.eur_usd import get_prices_for_day
from hydradx.apps.stableswap.eur_usd_arbitrage_sim import smooth_binance_with_kraken


def main() -> None:
    demo_day = date.today() - timedelta(days=1)
    binance_demo = get_prices_for_day("binance", demo_day)
    kraken_demo = get_prices_for_day("kraken", demo_day)
    demo = smooth_binance_with_kraken(binance_demo, kraken_demo)
    print(demo.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
