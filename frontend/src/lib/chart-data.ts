export async function fetchCurrencyData(currency: string) {
  const res = await fetch(`/api/dashboard?currency=${currency}`);
  if (!res.ok) throw new Error(`Failed to fetch ${currency} data`);
  return res.json();
}

export const CURRENCY_OPTIONS = ["usd", "eur", "jpy", "cny"] as const;

export const INDICATOR_COLORS = [
  { key: "SMA_5", color: "#FF6384" },
  { key: "EMA_5", color: "#FF9F40" },
  { key: "SMA_20", color: "#FFCD56" },
  { key: "EMA_20", color: "#4BC0C0" },
  { key: "SMA_60", color: "#36A2EB" },
  { key: "EMA_60", color: "#9966FF" },
  { key: "SMA_120", color: "#C9CBCF" },
  { key: "EMA_120", color: "#8DD1E1" },
] as const;
