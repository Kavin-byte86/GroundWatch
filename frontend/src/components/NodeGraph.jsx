import { useState, useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

const METRICS = [
  { key: 'tilt_deg', label: 'Tilt', unit: '°', color: '#60a5fa' },
  { key: 'vibration_rms', label: 'Vibration', unit: ' g', color: '#a78bfa' },
  { key: 'strain_mm', label: 'Strain', unit: ' mm', color: '#34d399' },
]

function formatTime(ts) {
  if (!ts) return ''
  try {
    const d = new Date(ts)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  } catch {
    return ''
  }
}

export default function NodeGraph({ selectedNodeId, nodes, nodeHistory }) {
  const [activeMetric, setActiveMetric] = useState(null) // null = show all three

  const node = selectedNodeId ? nodes[selectedNodeId] : null
  const history = selectedNodeId ? nodeHistory[selectedNodeId] || [] : []

  const chartData = useMemo(() => {
    return history.map((pt) => ({
      time: formatTime(pt.timestamp),
      tilt_deg: pt.tilt_deg,
      vibration_rms: pt.vibration_rms,
      strain_mm: pt.strain_mm,
    }))
  }, [history])

  if (!selectedNodeId) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-panel-muted p-4">
        Select a node on the map
      </div>
    )
  }

  const visibleMetrics = activeMetric
    ? METRICS.filter((m) => m.key === activeMetric)
    : METRICS

  return (
    <div className="p-3 h-full flex flex-col">
      {/* Node header */}
      <div className="flex items-center justify-between mb-2">
        <div className="text-sm font-semibold text-panel-heading">
          {selectedNodeId}
          {node && (
            <span className="ml-2 text-xs font-normal text-panel-muted">
              {node.status} · risk {node.risk_score}
            </span>
          )}
        </div>
        <div className="flex gap-1">
          <button
            onClick={() => setActiveMetric(null)}
            className={`px-2 py-0.5 text-xs rounded-sm border ${
              activeMetric === null
                ? 'border-panel-text text-panel-text'
                : 'border-panel-border text-panel-muted hover:text-panel-text'
            }`}
          >
            All
          </button>
          {METRICS.map((m) => (
            <button
              key={m.key}
              onClick={() => setActiveMetric(m.key)}
              className={`px-2 py-0.5 text-xs rounded-sm border ${
                activeMetric === m.key
                  ? 'border-panel-text text-panel-text'
                  : 'border-panel-border text-panel-muted hover:text-panel-text'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* Current values */}
      {node && (
        <div className="flex gap-4 mb-2 text-xs">
          {METRICS.map((m) => (
            <span key={m.key} style={{ color: m.color }}>
              {m.label}: {node[m.key]}
              {m.unit}
            </span>
          ))}
        </div>
      )}

      {/* Chart */}
      <div className="flex-1 min-h-0">
        {chartData.length === 0 ? (
          <div className="flex items-center justify-center h-full text-xs text-panel-muted">
            Awaiting data…
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: -12 }}>
              <CartesianGrid stroke="#2a2e3a" strokeDasharray="3 3" />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 10, fill: '#6b7280' }}
                tickLine={false}
                axisLine={{ stroke: '#2a2e3a' }}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fontSize: 10, fill: '#6b7280' }}
                tickLine={false}
                axisLine={{ stroke: '#2a2e3a' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#181b24',
                  border: '1px solid #2a2e3a',
                  borderRadius: 4,
                  fontSize: 11,
                }}
                labelStyle={{ color: '#e5e7eb' }}
              />
              {visibleMetrics.map((m) => (
                <Line
                  key={m.key}
                  type="monotone"
                  dataKey={m.key}
                  stroke={m.color}
                  strokeWidth={1.5}
                  dot={false}
                  name={`${m.label} (${m.unit.trim()})`}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  )
}
