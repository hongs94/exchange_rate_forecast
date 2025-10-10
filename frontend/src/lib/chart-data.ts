import { BASE_URL } from "@/constant";

export async function fetchCurrencyData(currency: string) {
  const res = await fetch(`${BASE_URL}/dashboard?currency=${currency}`);
  if (!res.ok) throw new Error(`Failed to fetch ${currency} data`);
  return res.json();
}

export const CURRENCY_OPTIONS = ["usd", "eur", "jpy", "cny"] as const;


export const INDICATOR_COLORS = [
  { key: "MA_5", color: "#FF6384" },
  { key: "EMA_5", color: "#FF9F40" },
  { key: "MA_20", color: "#FFCD56" },
  { key: "EMA_20", color: "#4BC0C0" },
  { key: "MA_60", color: "#36A2EB" },
  { key: "EMA_60", color: "#9966FF" },
  { key: "MA_120", color: "#C9CBCF" },
  { key: "EMA_120", color: "#8DD1E1" },
] as const;
