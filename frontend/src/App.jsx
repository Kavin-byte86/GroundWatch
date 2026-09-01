import { useState, useEffect, useCallback, useRef } from 'react'
import { subscribe, onConnectionChange, send } from './socket.js'
import MapView from './components/MapView.jsx'
import NodeGraph from './components/NodeGraph.jsx'
import ControlPanel from './components/ControlPanel.jsx'
import AlertFeed from './components/AlertFeed.jsx'
import { Wifi, WifiOff } from 'lucide-react'

const MAX_HISTORY_POINTS = 60 // keep last 60 data points per node for charts
const MAX_ALERTS = 200

export default function App() {
  const [nodes, setNodes] = useState({})
  const [nodeHistory, setNodeHistory] = useState({}) // { nodeId: [ { timestamp, tilt_deg, vibration_rms, strain_mm } ] }
  const [alerts, setAlerts] = useState([])
  const [selectedNodeId, setSelectedNodeId] = useState(null)
  const [connectionStatus, setConnectionStatus] = useState('disconnected')
  const [systemStatus, setSystemStatus] = useState({
    active_nodes: 0,
    excluded_nodes: 0,
    gateway_status: 'offline',
  })

  useEffect(() => {
    const unsubConn = onConnectionChange(setConnectionStatus)

    const unsubMsg = subscribe((data) => {
      if (data.type === 'node_update') {
        setNodes((prev) => ({
          ...prev,
          [data.node_id]: data,
        }))

        setNodeHistory((prev) => {
          const existing = prev[data.node_id] || []
          const point = {
            timestamp: data.timestamp,
            tilt_deg: data.tilt_deg,
            vibration_rms: data.vibration_rms,
            strain_mm: data.strain_mm,
          }
          const updated = [...existing, point]
          if (updated.length > MAX_HISTORY_POINTS) {
            updated.splice(0, updated.length - MAX_HISTORY_POINTS)
          }
          return { ...prev, [data.node_id]: updated }
        })
      } else if (data.type === 'alert') {
        setAlerts((prev) => {
          const next = [data, ...prev]
          if (next.length > MAX_ALERTS) next.length = MAX_ALERTS
          return next
        })
      } else if (data.type === 'system_status') {
        setSystemStatus({
          active_nodes: data.active_nodes,
          excluded_nodes: data.excluded_nodes,
          gateway_status: data.gateway_status,
        })
      }
    })

    return () => {
      unsubConn()
      unsubMsg()
    }
  }, [])

  const handleSend = useCallback((command) => {
    send(command)
  }, [])

  const connDotColor =
    connectionStatus === 'connected'
      ? 'bg-risk-safe'
      : connectionStatus === 'reconnecting'
        ? 'bg-risk-watch'
        : 'bg-panel-muted'

  const connLabel =
    connectionStatus === 'connected'
      ? 'Connected'
      : connectionStatus === 'reconnecting'
        ? 'Reconnecting…'
        : 'Disconnected'

  return (
    <div className="flex flex-col h-screen w-screen bg-panel-bg text-panel-text font-sans overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-2 border-b border-panel-border bg-panel-surface shrink-0">
        <h1 className="text-base font-semibold text-panel-heading tracking-tight">
          GroundWatch
        </h1>

        <div className="flex items-center gap-4 text-xs text-panel-muted">
          <span>
            Nodes: <span className="text-panel-text">{systemStatus.active_nodes}</span> active
            {systemStatus.excluded_nodes > 0 && (
              <span className="ml-1">
                / <span className="text-risk-warning">{systemStatus.excluded_nodes}</span> excluded
              </span>
            )}
          </span>
          <span>
            Gateway:{' '}
            <span
              className={
                systemStatus.gateway_status === 'online'
                  ? 'text-risk-safe'
                  : 'text-risk-critical'
              }
            >
              {systemStatus.gateway_status}
            </span>
          </span>
        </div>

        <div className="flex items-center gap-1.5 text-xs">
          <span className={`w-2 h-2 rounded-full ${connDotColor}`} />
          <span className="text-panel-muted">{connLabel}</span>
        </div>
      </header>

      {/* Main content */}
      <main className="flex flex-1 min-h-0">
        {/* Map — left/center 60% */}
        <section className="w-[60%] border-r border-panel-border">
          <MapView
            nodes={nodes}
            selectedNodeId={selectedNodeId}
            onSelectNode={setSelectedNodeId}
          />
        </section>

        {/* Right sidebar — 40% */}
        <aside className="w-[40%] flex flex-col min-h-0">
          <div className="flex-1 min-h-0 overflow-y-auto border-b border-panel-border">
            <NodeGraph
              selectedNodeId={selectedNodeId}
              nodes={nodes}
              nodeHistory={nodeHistory}
            />
          </div>
          <div className="shrink-0 overflow-y-auto" style={{ maxHeight: '45%' }}>
            <ControlPanel
              nodes={nodes}
              onSend={handleSend}
            />
          </div>
        </aside>
      </main>

      {/* Alert feed — bottom strip */}
      <footer className="h-40 shrink-0 border-t border-panel-border bg-panel-surface">
        <AlertFeed alerts={alerts} />
      </footer>
    </div>
  )
}
