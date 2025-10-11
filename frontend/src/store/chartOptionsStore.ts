import { create } from "zustand";

interface ChartOptionsState {
  model: string;
  setModel: (model: string) => void;
  currency: string;
  setCurrency: (currency: string) => void;
  period: "recent" | "all";
  setPeriod: (period: "recent" | "all") => void;
}

export const useChartOptionsStore = create<ChartOptionsState>((set) => ({
  model: "LSTM",
  setModel: (model) => set({ model }),
  currency: "usd",
  setCurrency: (currency) => set({ currency }),
  period: "recent",
  setPeriod: (period) => set({ period }),
}));