import { useState } from "react"
import { DashboardHeader } from "@/components/dashboard-header"
import { LeftSidebar } from "@/components/left-sidebar"
import { ChartArea } from "@/components/chart-area"
import { ChatSidebar } from "@/components/chat-sidebar"

export default function Dashboard() {
  const [isLeftSidebarOpen, setIsLeftSidebarOpen] = useState(true)
  const [isChatSidebarOpen, setIsChatSidebarOpen] = useState(true)

  return (
    <div className="flex h-screen w-full overflow-hidden">
      <LeftSidebar isOpen={isLeftSidebarOpen} />
      <div className="flex flex-col flex-1 min-w-0">
        <DashboardHeader
          onToggleLeftSidebar={() => setIsLeftSidebarOpen(!isLeftSidebarOpen)}
          onToggleChatSidebar={() => setIsChatSidebarOpen(!isChatSidebarOpen)}
        />
        <div className="flex flex-1 min-h-0 overflow-hidden">
          <ChartArea />
          <ChatSidebar isOpen={isChatSidebarOpen} />
        </div>
      </div>
    </div>
  )
}