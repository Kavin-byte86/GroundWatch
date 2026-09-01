import { useEffect, useRef } from 'react'
import { AlertTriangle, AlertCircle, Info } from 'lucide-react'

const SEVERITY_STYLES = {
  critical: {
    bg: 'bg-red-950/50',
    border: 'border-risk-critical',
    badge: 'bg-risk-critical text-white',
    label: 'CRITICAL',
  },
  warning: {
    bg: 'bg-orange-950/30',
    border: 'border-risk-warning',
    badge: 'bg-risk-warning text-black',
    label: 'WARNING',
  },
  watch: {
    bg: 'bg-yellow-950/20',
    border: 'border-risk-watch',
    badge: 'bg-risk-watch text-black',
    label: 'WATCH',
  },
  info: {
    bg: '',
    border: 'border-panel-border',
    badge: 'bg-panel-muted text-white',
    label: 'INFO',
  },
}

function SeverityIcon({ severity }) {
  const cls = 'w-3.5 h-3.5 shrink-0'
  switch (severity) {
    case 'critical':
      return <AlertCircle className={cls} style={{ color: '#ef4444' }} />
    case 'warning':
      return <AlertTriangle className={cls} style={{ color: '#f97316' }} />
    case 'watch':
      return <AlertTriangle className={cls} style={{ color: '#eab308' }} />
    default:
      return <Info className={cls} style={{ color: '#6b7280' }} />
  }
}

function formatRelativeTime(ts) {
  if (!ts) return ''
  try {
    const diff = Date.now() - new Date(ts).getTime()
    if (diff < 0) return 'just now'
    const secs = Math.floor(diff / 1000)
    if (secs < 60) return `${secs}s ago`
    const mins = Math.floor(secs / 60)
    if (mins < 60) return `${mins}m ago`
    const hrs = Math.floor(mins / 60)
    return `${hrs}h ago`
  } catch {
    return ''
  }
}

export default function AlertFeed({ alerts }) {
  const listRef = useRef(null)
  const isScrolledUpRef = useRef(false)

  // Track if user has scrolled up
  const handleScroll = () => {
    const el = listRef.current
    if (!el) return
    // If scrolled within 40px of top, consider it "at newest"
    isScrolledUpRef.current = el.scrollTop > 40
  }

  // Auto-scroll to top (newest) unless user has scrolled away
  useEffect(() => {
    if (!isScrolledUpRef.current && listRef.current) {
      listRef.current.scrollTop = 0
    }
  }, [alerts])

  if (alerts.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-xs text-panel-muted">
        No alerts
      </div>
    )
  }

  return (
    <div
      ref={listRef}
      onScroll={handleScroll}
      className="h-full overflow-y-auto alert-feed-scroll"
    >
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-panel-surface text-panel-muted border-b border-panel-border">
          <tr>
            <th className="text-left px-3 py-1.5 w-20">Severity</th>
            <th className="text-left px-3 py-1.5 w-16">Node</th>
            <th className="text-left px-3 py-1.5">Message</th>
            <th className="text-right px-3 py-1.5 w-20">Time</th>
          </tr>
        </thead>
        <tbody>
          {alerts.map((alert, idx) => {
            const style = SEVERITY_STYLES[alert.severity] || SEVERITY_STYLES.info
            return (
              <tr
                key={`${alert.timestamp}-${alert.node_id}-${idx}`}
                className={`border-b border-panel-border ${style.bg}`}
              >
                <td className="px-3 py-1.5">
                  <span className="flex items-center gap-1.5">
                    <SeverityIcon severity={alert.severity} />
                    <span
                      className={`px-1.5 py-0.5 rounded-sm text-[10px] font-semibold leading-none ${style.badge}`}
                    >
                      {style.label}
                    </span>
                  </span>
                </td>
                <td className="px-3 py-1.5 font-mono text-panel-text">{alert.node_id}</td>
                <td className="px-3 py-1.5 text-panel-text">{alert.message}</td>
                <td className="px-3 py-1.5 text-right text-panel-muted whitespace-nowrap">
                  {formatRelativeTime(alert.timestamp)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
