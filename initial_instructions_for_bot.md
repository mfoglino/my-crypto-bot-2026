# 🤖 Trading Bot v2 — Especificación Técnica

## 📌 CONTEXTO GENERAL

Tengo experiencia previa desarrollando un bot de trading en crypto (Binance Futures, ETH/USDT).  
El bot anterior era funcional pero básico, y no fue rentable de forma consistente.

### Características del bot anterior:
- Operaba futuros en Binance
- Basado en indicadores simples:
    - RSI
    - Supertrend
- Lógica sin contexto de mercado (no distinguía entre tendencia y lateralidad)
- Dependía parcialmente de servicios externos pagos (que se quieren eliminar)
- Infraestructura ya desarrollada (repo existente, ejecución automática, conexión al broker)

---

## 🎯 OBJETIVO

Diseñar una **nueva versión del bot**, más robusta y profesional, que:

- Sea completamente autónoma usando solo la API de Binance
- No dependa de servicios externos pagos
- Sea deployable en infraestructura simple (ej: AWS free tier)
- Sea rentable después de comisiones y slippage
- Evite sobreoperar y mejore la calidad de las señales

---

## ⚙️ RESTRICCIONES TÉCNICAS

- Fuente de datos:
    - Binance WebSocket (precio, trades, order book)
    - Binance REST API (órdenes, histórico)
- Lenguaje: Python
- Infraestructura:
    - AWS free tier o equivalente
    - CPU/RAM limitadas
- No HFT (no competir por latencia)
- Evitar dependencias externas innecesarias

---

## ♻️ REUTILIZACIÓN DEL BOT EXISTENTE

### Reutilizar:
- Conexión a Binance API
- Ejecución de órdenes
- Gestión de posiciones
- Logging básico

### NO reutilizar:
- Lógica de señales (debe rediseñarse completamente)

---

## 🧠 NUEVO ENFOQUE ESTRATÉGICO

El nuevo bot debe basarse en **contexto de mercado**, no en indicadores aislados.

---

## 1️⃣ DETECCIÓN DE RÉGIMEN DE MERCADO

Clasificar el mercado en:

- Tendencial
- Lateral

### Ejemplo de implementación:
- Usar ATR (Average True Range)
- Usar volatilidad
- Opcional: pendiente de media móvil

```python
if ATR > threshold:
    market_regime = "trend"
else:
    market_regime = "range"