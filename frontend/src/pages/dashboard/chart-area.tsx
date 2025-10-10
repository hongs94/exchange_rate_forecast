import { Card, CardContent } from "@/components/ui/card";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Legend,
  Tooltip,
  ChartData,
  ChartOptions,
} from "chart.js";
import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { useChartOptionsStore } from "@/store/chartOptionsStore";
import { fetchCurrencyData, INDICATOR_COLORS } from "@/lib/chart-data";

ChartJS.register(LineElement, PointElement, CategoryScale, LinearScale, Legend, Tooltip);

interface ChartAreaProps {
  visibleItems: Record<string, boolean>;
}

export function ChartArea({ visibleItems }: ChartAreaProps) {
  const { currency, model, period } = useChartOptionsStore();

  const { data, isLoading, isError } = useQuery({
    queryKey: ["dashboard", currency],
    queryFn: () => fetchCurrencyData(currency),
    staleTime: 1000 * 60 * 5,
  });

  const chartData = useMemo<ChartData<"line">>(() => {
    if (!data) return { labels: [], datasets: [] };

    const allDatesSet = new Set<string>();
    data.real_prices.forEach((p: any) => allDatesSet.add(p.date));
    data.predicted_prices.forEach((p: any) => allDatesSet.add(p.date));
    const allDates = Array.from(allDatesSet).sort();
    const filteredDates = period === "recent" ? allDates.slice(-30) : allDates;

    const datasets: any[] = [];

    // 실제 가격
    const realMap = new Map(data.real_prices.map((p: any) => [p.date, p[currency]]));
    datasets.push({
      label: `${currency.toUpperCase()} - Real`,
      data: filteredDates.map((d) => realMap.get(d) ?? null),
      borderColor: "#36A2EB",
      backgroundColor: "transparent",
      tension: 0.3,
      pointRadius: 0,
      hidden: visibleItems[`${currency.toUpperCase()} - Real`] === false,
    });

    // SMA/EMA
    INDICATOR_COLORS.forEach((ind) => {
      const map = new Map(data.real_prices.map((p: any) => [p.date, p[ind.key]]));
      datasets.push({
        label: ind.key,
        data: filteredDates.map((d) => map.get(d) ?? null),
        borderColor: ind.color,
        backgroundColor: "transparent",
        tension: 0.3,
        pointRadius: 0,
        borderDash: [5, 5],
        hidden: visibleItems[ind.key] === false,
      });
    });

    // 예측값
    const predictedMap = new Map(data.predicted_prices.map((p: any) => [p.date, p[model]]));
    datasets.push({
      label: `Predicted (${model})`,
      data: filteredDates.map((d) => predictedMap.get(d) ?? null),
      borderColor: "#ef4444",
      backgroundColor: "#ef4444",
      borderDash: [5, 5],
      tension: 0.3,
      pointRadius: 7,
      pointStyle: "triangle",
      hidden: visibleItems[`Predicted (${model})`] === false,
    });

    return { labels: filteredDates, datasets };
  }, [data, currency, model, period, visibleItems]);

  const options: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: { legend: { display: false }, tooltip: { enabled: true } },
    scales: { x: { ticks: { maxRotation: 45, minRotation: 0 } }, y: { beginAtZero: false } },
  };

  return (
    <div className="flex-1 p-4 min-w-0 overflow-hidden">
      <Card className="h-full">
        <CardContent className="h-full p-4 flex items-center justify-center">
          {isLoading ? (
            <p>Loading chart...</p>
          ) : isError ? (
            <p>Failed to load data.</p>
          ) : (
            <Line data={chartData} options={options} />
          )}
        </CardContent>
      </Card>
    </div>
  );
}
