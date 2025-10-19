import { useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useChartOptionsStore } from "@/store/chartOptionsStore";
import { useQuery } from "@tanstack/react-query";
import { fetchCurrencyData, INDICATOR_COLORS, CURRENCY_OPTIONS } from "@/lib/chart-data";

interface LeftSidebarProps {
  isOpen: boolean;
  visibleItems: Record<string, boolean>;
  setVisibleItems: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
}

export function LeftSidebar({ isOpen, visibleItems, setVisibleItems }: LeftSidebarProps) {
  const { model, setModel, currency, setCurrency, period, setPeriod } = useChartOptionsStore();

  const { data } = useQuery({
    queryKey: ["dashboard", currency],
    queryFn: () => fetchCurrencyData(currency),
    staleTime: 1000 * 60 * 5,
  });

  // visibleItems 초기화
  useEffect(() => {
    if (!data) return;
    const items = [
      `${currency.toUpperCase()} - Real`,
      `${model}`,
      ...INDICATOR_COLORS.map((i) => i.key),
    ];
    const obj: Record<string, boolean> = {};
    items.forEach((i) => {
      obj[i] = visibleItems[i] ?? true; // 기존 상태 유지, 없으면 true
    });
    setVisibleItems(obj);
  }, [data, currency, model]);

  const toggleItem = (label: string) => {
    setVisibleItems((prev) => ({ ...prev, [label]: !prev[label] }));
  };

  return (
    <aside
      className={`bg-card border-r border-border transition-all duration-300 flex-shrink-0 h-full ${isOpen ? "w-64" : "w-0 overflow-hidden"
        }`}
    >
      <div className="p-4 space-y-4 w-64 h-full overflow-y-auto">
        <h3 className="font-medium text-foreground">차트 설정</h3>

        <div className="space-y-4">
          {/* 모델 선택 */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">예측 모델 선택</label>
            <Select value={model} onValueChange={setModel}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="예측 모델 선택" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="XGBoost">XGBoost</SelectItem>
                <SelectItem value="LSTM">LSTM</SelectItem>
                <SelectItem value="LSTM_Attention">LSTM_Attention</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* 통화 선택 */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">통화 선택</label>
            <Select value={currency} onValueChange={setCurrency}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="통화 선택" />
              </SelectTrigger>
              <SelectContent>
                {CURRENCY_OPTIONS.map((cur) => (
                  <SelectItem key={cur} value={cur}>
                    {cur.toUpperCase()}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* 기간 선택 */}
          <div className="space-y-2 pt-2">
            <label className="text-sm font-medium text-foreground">기간 선택</label>
            <div className="grid grid-cols-2 gap-2">
              <Button
                variant={period === "recent" ? "secondary" : "outline"}
                onClick={() => setPeriod("recent")}
              >
                6개월
              </Button>
              <Button
                variant={period === "all" ? "secondary" : "outline"}
                onClick={() => setPeriod("all")}
              >
                전체
              </Button>
            </div>
          </div>

          {/* 체크박스 범례 */}
          {data && Object.keys(visibleItems).length > 0 && (
            <Card className="mt-4 p-2">
              <CardContent className="flex flex-col gap-2">
                {[
                  { label: `${currency.toUpperCase()}`, color: "#36A2EB" },
                  { label: `${model}`, color: "#ef4444" },
                  ...INDICATOR_COLORS.map((i) => ({ label: i.key, color: i.color })),
                ].map((item) => (
                  <label key={item.label} className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={visibleItems[item.label] ?? true} // 기본값 true
                      onChange={() => toggleItem(item.label)}
                    />
                    <span className="w-4 h-4 rounded" style={{ backgroundColor: item.color }} />
                    <span className="text-xs">{item.label}</span>
                  </label>
                ))}
              </CardContent>
            </Card>
          )}

        </div>
      </div>
    </aside>
  );
}
