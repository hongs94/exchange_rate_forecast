import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useChartOptionsStore } from "@/store/chartOptionsStore"
import { Button } from "@/components/ui/button"

interface LeftSidebarProps {
  isOpen: boolean
}

export function LeftSidebar({ isOpen }: LeftSidebarProps) {
  const { model, setModel, currency, setCurrency, period, setPeriod } = useChartOptionsStore()

  return (
    <aside
      className={`bg-card border-r border-border transition-all duration-300 flex-shrink-0 h-full ${isOpen ? "w-64" : "w-0 overflow-hidden"
        }`}
    >
      <div className="p-4 space-y-4 w-64 h-full overflow-y-auto">
        <h3 className="font-medium text-foreground">차트 설정</h3>
        <div className="space-y-4">
          <div className="space-y-3">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">모델 선택</label>
              <Select value={model} onValueChange={setModel}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="모델 선택" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="LSTM-Rolling">LSTM-Rolling</SelectItem>
                  {/* <SelectItem value="Attention_LSTM-Rolling">Attention_LSTM-Rolling</SelectItem> */}
                  <SelectItem value="XGBoost-Rolling">XGBoost-Rolling</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">통화 선택</label>
              <Select value={currency} onValueChange={setCurrency}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="통화 선택" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="usd">USD (달러)</SelectItem>
                  <SelectItem value="eur">EUR (유로)</SelectItem>
                  <SelectItem value="cny">CNY (위안)</SelectItem>
                  <SelectItem value="jpy">JPY (엔)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2 pt-2">
              <label className="text-sm font-medium text-foreground">기간 선택</label>
              <div className="grid grid-cols-2 gap-2">
                <Button
                  variant={period === "recent" ? "secondary" : "outline"}
                  onClick={() => setPeriod("recent")}
                >
                  최근
                </Button>
                <Button
                  variant={period === "all" ? "secondary" : "outline"}
                  onClick={() => setPeriod("all")}
                >
                  전체
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </aside>
  )
}