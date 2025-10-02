import { useChartOptionsStore } from "@/store/chartOptionsStore";
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
import { useMemo, type JSX, useState, useEffect } from "react";

ChartJS.register(LineElement, PointElement, CategoryScale, LinearScale, Legend, Tooltip);

interface ApiDataPoint {
  index: string;
  [key: string]: number | string;
}
interface ApiPredictionPoint {
  model: string;
  date: string;
  [key: string]: number;
}
interface ApiResponse {
  data: ApiDataPoint[];
  predicted_price: ApiPredictionPoint[];
}
interface ProcessedData {
  actual: { date: string; value: number }[];
  models: { [model: string]: { date: string; value: number } };
}

export function ChartArea(): JSX.Element {
  const { currency, model, period } = useChartOptionsStore();

  const [rawData, setRawData] = useState<ProcessedData | null>(null);
  const [displayData, setDisplayData] = useState<ProcessedData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!currency) {
      setRawData(null);
      return;
    }
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`http://localhost:8000/dashboard?currency=${currency}`);
        if (!response.ok) throw new Error("서버에서 데이터를 가져오는 데 실패했습니다.");
        const json: ApiResponse = await response.json();
        const actual = json.data.map((d) => ({ date: d.index, value: d[currency] as number }));
        const models: ProcessedData["models"] = {};
        json.predicted_price.forEach((p) => {
          models[p.model] = { date: p.date, value: p[currency] as number };
        });
        setRawData({ actual, models });
      } catch (e) {
        setError(e instanceof Error ? e.message : "알 수 없는 오류가 발생했습니다.");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [currency]);

  useEffect(() => {
    if (!rawData || !rawData.actual.length) {
      setDisplayData(null);
      return;
    }

    let filteredActual = rawData.actual;

    if (period === "recent") {
      const lastDate = new Date(rawData.actual[rawData.actual.length - 1].date);
      const oneYearAgo = new Date(lastDate);
      oneYearAgo.setFullYear(lastDate.getFullYear() - 1);
      const oneYearAgoString = oneYearAgo.toISOString().split('T')[0];
      filteredActual = rawData.actual.filter(d => d.date >= oneYearAgoString);
    }
    setDisplayData({ actual: filteredActual, models: rawData.models });
  }, [rawData, period]);

  const chartData: ChartData<"line"> | null = useMemo(() => {
    if (!displayData?.actual?.length || !model || !displayData.models[model]) {
      return null;
    }
    const actualData = displayData.actual;
    const selectedModelData = displayData.models[model];
    const predictionDate = selectedModelData.date;
    const labels = [...actualData.map((d) => d.date)];
    if (actualData.length > 0 && !labels.includes(predictionDate)) {
      labels.push(predictionDate);
    }
    const datasets = [
      {
        label: "실제값",
        data: actualData.map((d) => d.value),
        borderColor: "#6366f1",
        tension: 0.3,
        pointRadius: 1,
      },
    ];
    if (actualData.length > 0) {
        const lastActualValue = actualData[actualData.length - 1].value;
        const predictionPoints = Array(actualData.length - 1).fill(null);
        predictionPoints.push(lastActualValue);
        predictionPoints.push(selectedModelData.value);
        datasets.push({
            label: `${model} 예측값`,
            data: predictionPoints,
            borderColor: "#f59e42",
            borderDash: [6, 6],
            tension: 0.3,
            pointRadius: (context) => (context.dataIndex === labels.length - 1 ? 5 : 0),
            pointBackgroundColor: "#f59e42",
        });
    }
    return { labels, datasets };
  }, [displayData, model]);

  const options: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: true },
      tooltip: { mode: "index", intersect: false },
    },
    scales: {
      x: { title: { display: true, text: "날짜" } },
      y: { title: { display: true, text: `환율 (${currency?.toUpperCase()})` } },
    },
    interaction: {
      mode: 'index',
      intersect: false,
    },
  };

  const renderContent = () => {
    if (loading) return <div className="text-muted-foreground">데이터를 불러오는 중...</div>;
    if (error) return <div className="text-red-500">에러: {error}</div>;
    if (!currency || !model) return <div className="text-muted-foreground">통화와 모델을 선택하세요.</div>;
    if (chartData) return <div className="w-full h-full"><Line data={chartData} options={options} /></div>;
    return <div className="text-muted-foreground">선택한 조건의 데이터가 없습니다.</div>;
  };

  return (
    <div className="flex-1 p-4 min-w-0 overflow-hidden">
      <Card className="h-full">
        <CardContent className="h-full p-4 flex items-center justify-center">
          {renderContent()}
        </CardContent>
      </Card>
    </div>
  );
}