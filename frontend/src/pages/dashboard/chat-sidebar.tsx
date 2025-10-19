import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Send } from "lucide-react"

interface ChatSidebarProps {
  isOpen: boolean
}

interface ChatMessage {
  sender: string
  content: string
  time: string
  isMe?: boolean
}

export function ChatSidebar({ isOpen }: ChatSidebarProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [wsConnected, setWsConnected] = useState(true)
  const ws = useRef<WebSocket | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    ws.current = new WebSocket("/ws/chat")
    ws.current.onopen = () => setWsConnected(true)
    ws.current.onclose = () => setWsConnected(false)
    ws.current.onerror = () => setWsConnected(false)
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setMessages(prev => [
        ...prev,
        {
          sender: data.username,
          content: data.message,
          time: data.time ? new Date(data.time).toLocaleTimeString() : new Date().toLocaleTimeString(),
          isMe: data.isMe
        },
      ])
    }
    return () => {
      ws.current?.close()
    }
  }, [])

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleSend = () => {
    const trimmed = input.trim()
    if (!trimmed || !ws.current || ws.current.readyState !== WebSocket.OPEN) {
      setInput("")
      return
    }
    ws.current.send(trimmed)
    setInput("")
  }

  return (
    <aside
      className={`bg-card border-l border-border transition-all duration-300 flex flex-col flex-shrink-0 h-full ${isOpen ? "w-80" : "w-0 overflow-hidden"
        }`}
    >
      <div className="w-80 h-full flex flex-col">
        <div className="p-4 border-b border-border flex-shrink-0">
          <h3 className="font-medium text-foreground">채팅기록</h3>
        </div>

        <div className="flex-1 p-4 overflow-hidden">
          <ScrollArea className="h-full pr-4">
            <div className="space-y-3">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`p-3 rounded-lg ${msg.isMe
                    ? "bg-primary text-primary-foreground ml-8"
                    : "bg-accent"
                    }`}
                >
                  <p className={`text-xs font-medium ${msg.isMe ? "opacity-80" : "text-accent-foreground/80"} mb-1`}>
                    {msg.sender}
                  </p>
                  <p className={`text-sm ${msg.isMe ? "" : "text-accent-foreground"}`}>{msg.content}</p>
                  <span className={`text-xs ${msg.isMe ? "opacity-70" : "text-accent-foreground/70"}`}>{msg.time}</span>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>
        </div>

        <div className="p-4 border-t border-border space-y-3 flex-shrink-0">
          {!wsConnected && (
            <div className="text-red-500 text-sm mb-2 text-center">
              채팅을 이용하려면 로그인이 필요합니다.
            </div>
          )}
          <Textarea
            placeholder="메시지를 입력하세요..."
            className="min-h-[80px] resize-none"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
            disabled={!wsConnected}
          />
          <Button className="w-full" size="sm" onClick={handleSend} disabled={!wsConnected}>
            <Send className="w-4 h-4 mr-2" />
            전송
          </Button>
        </div>
      </div>
    </aside>
  )
}