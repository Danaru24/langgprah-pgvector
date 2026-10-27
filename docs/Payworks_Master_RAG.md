# Payworks — Documentación Técnica Maestra para RAG

> **PLANTILLA ILUSTRATIVA.** Reemplace este archivo por la versión oficial de `Payworks_Master_RAG.md` (~20,000 caracteres). El contenido siguiente existe solo para validar el pipeline de punta a punta: umbrales, SLOs, códigos y nombres de componentes son **de ejemplo** y no constituyen documentación oficial. Si cambia los encabezados, actualice `docs/golden_dataset.json`.

## 1. Introducción y Alcance

Payworks es la pasarela de pagos que permite a los comercios afiliados procesar cobros con tarjetas de crédito y débito en canales no presenciales (comercio electrónico, aplicaciones móviles y centros de atención telefónica). Este documento describe los conceptos de negocio, la arquitectura de integración, el flujo transaccional de extremo a extremo y las prácticas de monitoreo y confiabilidad (SRE) que deben seguir los equipos de desarrollo y operación.

El público objetivo son desarrolladores de integración, arquitectos de soluciones, analistas de soporte de segundo nivel e ingenieros de confiabilidad. Se asume familiaridad con HTTPS, formatos clave-valor y conceptos básicos de adquirencia de tarjetas.

## 2. Conceptos y Glosario

Esta sección define los términos que aparecen a lo largo de la integración. Los nombres de parámetros se escriben en mayúsculas porque así se envían en las solicitudes HTTP a la pasarela.

### 2.1 Afiliación y Merchant ID

La afiliación es el contrato mediante el cual el banco adquirente habilita a un comercio para aceptar pagos con tarjeta. Como resultado de la afiliación se asigna un identificador único llamado **MERCHANT_ID**, que acompaña a todas las transacciones y permite al adquirente determinar a qué cuenta se liquidan los fondos, qué comisiones aplican y qué giro comercial tiene el negocio.

Junto con el MERCHANT_ID se entregan credenciales de acceso (**USER** y **PASSWORD**) asociadas a un usuario de integración y, cuando aplica, un identificador de terminal virtual (**TERMINAL_ID**). Un mismo comercio puede tener varias afiliaciones, por ejemplo una para ventas en una sola exhibición y otra para meses sin intereses. Las credenciales nunca deben incrustarse en el código fuente; se almacenan en un gestor de secretos y se rotan periódicamente.

### 2.2 Tipos de Transacción (CMD_TRANS)

El parámetro **CMD_TRANS** indica la operación solicitada a la pasarela:

- **AUTH (Autorización directa):** reserva y cobra el importe en un solo paso. Es la operación más común en comercio electrónico cuando el bien o servicio se entrega de inmediato.
- **PREAUTH (Preautorización):** solo reserva el importe en la línea de crédito del tarjetahabiente sin generar el cargo definitivo. Se usa cuando el monto final puede variar, por ejemplo en hoteles o renta de autos.
- **POSTAUTH (Postautorización):** confirma una preautorización previa y genera el cargo definitivo por un importe igual o menor al reservado.
- **REVERSAL (Reverso):** anula técnicamente una transacción cuya respuesta no se recibió o que quedó en estado incierto.
- **VOID (Cancelación):** anula una transacción aprobada el mismo día, antes del cierre de lote.
- **REFUND (Devolución):** reintegra total o parcialmente el importe de una venta ya liquidada.

La diferencia clave entre AUTH y PREAUTH es el momento del cargo: AUTH cobra inmediatamente, mientras que PREAUTH bloquea fondos y requiere un POSTAUTH posterior para completar el cobro.

### 2.3 Modos de Operación (MODE)

El parámetro **MODE** define el comportamiento de la pasarela ante la solicitud y permite probar la integración sin afectar tarjetas reales:

- **PRD:** producción. La transacción se envía al autorizador y afecta la cuenta del tarjetahabiente.
- **AUT:** prueba en la que la pasarela siempre responde como aprobada.
- **DEC:** prueba en la que la pasarela siempre responde como declinada.
- **RND:** prueba con respuesta aleatoria (aprobada o declinada), útil para validar el manejo de ambos escenarios.

Antes de pasar a producción, el comercio debe certificar su integración ejecutando la matriz de pruebas en modos AUT, DEC y RND.

### 2.4 Códigos de Resultado (RESULTADO_PAYW)

La respuesta de la pasarela incluye el campo **RESULTADO_PAYW**, que resume el desenlace de la operación:

- **A — Aprobada:** el emisor autorizó la transacción; se devuelve un **CODIGO_AUT** de seis caracteres y una **REFERENCIA** única.
- **D — Declinada:** el emisor rechazó la transacción (fondos insuficientes, tarjeta bloqueada, sospecha de fraude). El campo TEXTO amplía el motivo.
- **R — Rechazada:** la pasarela rechazó la solicitud antes de enviarla al emisor por errores de validación, credenciales inválidas o parámetros faltantes.
- **T — Sin respuesta del autorizador:** se agotó el tiempo de espera hacia el emisor. El estado de la transacción es incierto y debe tratarse según el runbook de timeouts y reversos automáticos.

### 2.5 3D Secure y Autenticación del Tarjetahabiente

3D Secure es el protocolo de autenticación del tarjetahabiente para compras no presenciales. Antes de la autorización, el comercio redirige al cliente al servicio de autenticación de su banco emisor, que valida su identidad mediante un código dinámico, biometría o la aplicación bancaria. El resultado se expresa con el indicador **ECI** y un criptograma de autenticación (**CAVV**) que se envían junto con la solicitud AUTH.

El beneficio principal para el comercio es la transferencia de responsabilidad (*liability shift*): ante un contracargo por fraude en una transacción autenticada, la responsabilidad recae en el emisor. Además, reduce la tasa de fraude y es obligatorio para ciertos giros de alto riesgo.

## 3. Arquitectura de la Solución

### 3.1 Componentes Principales

La integración de referencia se compone de los siguientes elementos:

1. **Frontend de checkout:** captura los datos de pago mediante un formulario alojado o campos tokenizados, de modo que el número de tarjeta nunca toca los servidores del comercio.
2. **API de pagos del comercio (Payment Service):** microservicio desplegado en OpenShift que orquesta la transacción, aplica reglas de negocio e idempotencia y persiste el estado.
3. **Conector Payworks:** cliente HTTPS que construye la solicitud con MERCHANT_ID, USER, PASSWORD, CMD_TRANS y MODE, y traduce la respuesta a un modelo interno.
4. **Servicio 3D Secure (MPI):** gestiona la autenticación del tarjetahabiente y devuelve ECI y CAVV.
5. **Base de datos transaccional:** registra cada intento con su REFERENCIA, estado y marcas de tiempo; es la fuente de verdad para conciliación.
6. **Bus de eventos:** publica eventos de pago aprobado, declinado o revertido para sistemas de órdenes, facturación y antifraude.
7. **Observabilidad:** métricas en Prometheus, trazas distribuidas y logs centralizados.

### 3.2 Seguridad y Cumplimiento PCI DSS

Toda la comunicación con la pasarela se realiza sobre TLS 1.2 o superior con validación de certificado. Los datos de tarjeta se tokenizan en el navegador, por lo que el alcance PCI DSS del comercio se reduce al nivel SAQ A-EP o SAQ A según el método de captura.

Controles obligatorios: no almacenar el código de seguridad (CVV2) bajo ninguna circunstancia; enmascarar el número de tarjeta en logs mostrando solo los últimos cuatro dígitos; cifrar en reposo los datos sensibles; restringir por red el acceso al conector mediante NetworkPolicies; y rotar credenciales cada 90 días. Los secretos se montan desde un gestor de secretos y nunca se exponen como variables en texto plano dentro de los manifiestos.

## 4. Flujo Transaccional

### 4.1 Flujo de Autorización (AUTH)

El flujo de una autorización directa sigue estos pasos:

1. El cliente confirma su compra en el frontend de checkout y los datos de tarjeta se tokenizan en el navegador.
2. El frontend envía el token y el importe al Payment Service, junto con una llave de idempotencia generada por el cliente.
3. El Payment Service valida la orden, verifica que la llave de idempotencia no haya sido procesada y registra la transacción en estado **PENDIENTE**.
4. Si el comercio exige 3D Secure, se invoca al MPI; el cliente completa la autenticación y se obtienen ECI y CAVV.
5. El conector Payworks construye la solicitud con CMD_TRANS=AUTH, MODE=PRD, MERCHANT_ID, credenciales, importe, token y datos de autenticación, y la envía por HTTPS con un timeout de 30 segundos.
6. La pasarela valida la solicitud, la enruta a la red de la marca y esta al banco emisor, que decide aprobar o declinar.
7. La pasarela responde con RESULTADO_PAYW, CODIGO_AUT, REFERENCIA y TEXTO.
8. El Payment Service actualiza el estado a **APROBADA**, **DECLINADA** o **RECHAZADA**, publica el evento correspondiente en el bus y responde al frontend.
9. Si RESULTADO_PAYW es T o la conexión expira, la transacción pasa a **INCIERTA** y se dispara el procedimiento de reverso automático.

Consideraciones de diseño: la llave de idempotencia evita cobros duplicados cuando el cliente reintenta; cada paso emite una traza con la REFERENCIA como atributo para correlacionar logs; y el estado PENDIENTE nunca debe permanecer más de cinco minutos sin resolución, de lo contrario un proceso de barrido lo concilia contra la pasarela.

Los reintentos automáticos solo se permiten ante errores de conexión previos al envío (por ejemplo, fallo de DNS o rechazo de conexión TCP). Una vez enviada la solicitud, nunca se reintenta una AUTH: se aplica el reverso para evitar dobles cargos.

### 4.2 Preautorización y Postautorización

La preautorización (CMD_TRANS=PREAUTH) bloquea el importe en la tarjeta sin cobrarlo. El comercio conserva la REFERENCIA devuelta y, cuando conoce el monto final, envía una postautorización (CMD_TRANS=POSTAUTH) con esa REFERENCIA para generar el cargo definitivo. El importe de la postautorización puede ser igual o menor al preautorizado, nunca mayor.

Si la postautorización no se realiza dentro de la vigencia definida por el emisor (comúnmente entre 7 y 30 días), el bloqueo se libera automáticamente y el comercio pierde la posibilidad de cobrar. Por ello se recomienda un proceso programado que alerte sobre preautorizaciones próximas a vencer.

### 4.3 Reversos y Cancelaciones

El reverso (CMD_TRANS=REVERSAL) es una operación técnica que se usa cuando el comercio no sabe con certeza si la transacción fue aprobada, típicamente por un timeout o una respuesta T. El reverso garantiza que, si el emisor sí aprobó, los fondos se liberen sin generar cargo.

La cancelación (CMD_TRANS=VOID) es una operación de negocio sobre una transacción **aprobada y confirmada**, solicitada el mismo día antes del cierre de lote, por ejemplo cuando el cliente se arrepiente de la compra. Regla práctica: ante incertidumbre técnica se usa reverso; ante una decisión de negocio sobre una venta conocida del día se usa cancelación.

### 4.4 Devoluciones (REFUND)

La devolución (CMD_TRANS=REFUND) reintegra al tarjetahabiente el importe de una venta ya liquidada, total o parcialmente. Requiere la REFERENCIA de la transacción original y el importe a devolver, que no puede exceder el monto pendiente de devolución. El reflejo en el estado de cuenta del cliente depende del emisor y puede tardar varios días hábiles.

## 5. Monitoreo y SRE

### 5.1 SLIs y SLOs

Los indicadores de nivel de servicio (SLIs) del flujo de pagos son: disponibilidad (porcentaje de solicitudes con respuesta distinta de error 5xx o timeout), latencia de autorización (tiempo entre la recepción en el Payment Service y la respuesta al frontend) y tasa de transacciones inciertas.

Los objetivos (SLOs) acordados son: disponibilidad mensual de **99.95%**; latencia **p95 menor a 2.5 segundos** y p99 menor a 5 segundos para AUTH; y tasa de transacciones inciertas menor a 0.1% del volumen diario. El presupuesto de error mensual derivado de la disponibilidad equivale a unos 21 minutos.

### 5.2 Métricas y Dashboards

El dashboard principal de pagos debe mostrar, como mínimo: transacciones por segundo por tipo (CMD_TRANS); distribución de RESULTADO_PAYW (A, D, R, T) en porcentaje; latencias p50, p95 y p99 hacia la pasarela; tasa de errores HTTP del conector; número de transacciones en estado PENDIENTE e INCIERTA; reversos automáticos ejecutados por hora; y consumo del presupuesto de error (*burn rate*).

Las métricas se etiquetan por MERCHANT_ID y CMD_TRANS para aislar problemas de un comercio o una operación específica.

### 5.3 Alertamiento

Las alertas se definen sobre síntomas visibles para el cliente:

- **Crítica:** tasa de RESULTADO_PAYW=R o T superior al **5% durante 5 minutos**, o burn rate del presupuesto de error mayor a 14x en una hora.
- **Alta:** latencia p95 superior a 2.5 segundos durante 10 minutos.
- **Media:** tasa de declinadas (D) superior al doble de su media semanal durante 30 minutos, posible señal de fraude o problema de un emisor.
- **Informativa:** preautorizaciones próximas a vencer y fallos en el proceso de barrido.

Toda alerta crítica o alta debe enlazar al runbook correspondiente y enviarse a la guardia de SRE.

### 5.4 Runbook: Timeouts y Reversos Automáticos

**Síntoma:** la pasarela no responde dentro de los 30 segundos del timeout, o responde con RESULTADO_PAYW=T. La transacción queda en estado INCIERTA.

**Acción automática:** el Payment Service encola un reverso (CMD_TRANS=REVERSAL) con la REFERENCIA o los datos de la transacción original. El reverso se reintenta con retroceso exponencial (1, 2, 4, 8 y 16 segundos) hasta cinco veces. Si el reverso es exitoso, la transacción pasa a **REVERTIDA** y se notifica al cliente que el cargo no se realizó. Si todos los intentos fallan, la transacción pasa a **REQUIERE_CONCILIACION** y se genera un ticket.

**Diagnóstico manual (guardia SRE):**

1. Revisar en el dashboard si el aumento de timeouts afecta a todos los comercios o a un solo MERCHANT_ID.
2. Verificar la latencia de red y resolución DNS hacia la pasarela desde los pods del conector.
3. Consultar la página de estado del proveedor y el canal de comunicación con el adquirente.
4. Comprobar que el pool de conexiones HTTP del conector no esté agotado ni haya hilos bloqueados.
5. Si la causa es externa, activar el modo degradado: mostrar al cliente un mensaje de reintento posterior y pausar campañas que generen picos de tráfico.

**Prohibido:** reintentar una AUTH después de un timeout, ya que puede generar doble cargo. Siempre se debe revertir primero.

**Cierre:** documentar la ventana del incidente, el número de transacciones revertidas y las que requirieron conciliación manual contra el reporte de liquidación del día siguiente.

### 5.5 Gestión de Incidentes y Escalamiento

Los incidentes se clasifican por severidad:

- **SEV1:** imposibilidad total de cobrar o tasa de error superior al 25%. Se escala de inmediato al líder de guardia, al gerente de plataforma y al contacto técnico del adquirente; se abre puente de comunicación en menos de 15 minutos.
- **SEV2:** degradación parcial (un tipo de transacción o un grupo de comercios). Se escala al líder de guardia en menos de 30 minutos.
- **SEV3:** impacto menor sin afectación a cobros. Se atiende en horario laboral.

Todo incidente SEV1 o SEV2 requiere un análisis postmortem sin culpables en un plazo de cinco días hábiles, con acciones correctivas asignadas y fecha compromiso.
