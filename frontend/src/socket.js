const WS_URL = 'ws://localhost:8000/ws'
const RECONNECT_DELAY_MS = 3000

let ws = null
let listeners = new Set()
let connectionListeners = new Set()
let reconnectTimer = null
let currentStatus = 'disconnected' // 'connected' | 'disconnected' | 'reconnecting'

function notifyConnection(status) {
  currentStatus = status
  connectionListeners.forEach((fn) => fn(status))
}

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return
  }

  try {
    ws = new WebSocket(WS_URL)
  } catch {
    notifyConnection('reconnecting')
    scheduleReconnect()
    return
  }

  ws.onopen = () => {
    notifyConnection('connected')
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
  }

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      listeners.forEach((fn) => fn(data))
    } catch {
      // Ignore malformed messages
    }
  }

  ws.onclose = () => {
    notifyConnection('reconnecting')
    scheduleReconnect()
  }

  ws.onerror = () => {
    // onclose will fire after onerror, so reconnect is handled there
  }
}

function scheduleReconnect() {
  if (reconnectTimer) return
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null
    connect()
  }, RECONNECT_DELAY_MS)
}

/** Subscribe to incoming WS messages. Returns an unsubscribe function. */
export function subscribe(fn) {
  listeners.add(fn)
  return () => listeners.delete(fn)
}

/** Subscribe to connection status changes. Returns an unsubscribe function. */
export function onConnectionChange(fn) {
  connectionListeners.add(fn)
  // Immediately notify with current status
  fn(currentStatus)
  return () => connectionListeners.delete(fn)
}

/** Send a JSON command to the server. Silently drops if not connected. */
export function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj))
  }
}

/** Get the current connection status. */
export function getConnectionStatus() {
  return currentStatus
}

// Auto-connect on module load
connect()
