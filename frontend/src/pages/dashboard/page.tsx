import { useState } from "react";
import { DashboardHeader } from "@/pages/dashboard/dashboard-header";
import { LeftSidebar } from "@/pages/dashboard/left-sidebar";
import { ChartArea } from "@/pages/dashboard/chart-area";
import { ChatSidebar } from "@/pages/dashboard/chat-sidebar";

export default function Dashboard() {
  const [isLeftSidebarOpen, setIsLeftSidebarOpen] = useState(true);
  const [isChatSidebarOpen, setIsChatSidebarOpen] = useState(true);

  // 체크박스 ON/OFF 상태
  const [visibleItems, setVisibleItems] = useState<Record<string, boolean>>({});

  return (
    <div className="flex h-screen w-full overflow-hidden">
      <LeftSidebar
        isOpen={isLeftSidebarOpen}
        visibleItems={visibleItems}
        setVisibleItems={setVisibleItems}
      />
      <div className="flex flex-col flex-1 min-w-0">
        <DashboardHeader
          onToggleLeftSidebar={() => setIsLeftSidebarOpen(!isLeftSidebarOpen)}
          onToggleChatSidebar={() => setIsChatSidebarOpen(!isChatSidebarOpen)}
        />
        <div className="flex flex-1 min-h-0 overflow-hidden">
          <ChartArea visibleItems={visibleItems} />
          <ChatSidebar isOpen={isChatSidebarOpen} />
        </div>
      </div>
    </div>
  );
}
