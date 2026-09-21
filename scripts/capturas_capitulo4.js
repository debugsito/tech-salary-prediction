// Capturas del prototipo para la sección 4.5 del documento (implementación).
//
// Complementa a capturas_anexo_k.js: aquí van la abstención sobre Perú, el
// visor de la respuesta JSON, el «acerca de» y la documentación interactiva
// de la interfaz de programación.
//
// Uso:  npm install puppeteer-core   (donde sea; pasar NODE_PATH)
//       node scripts/capturas_capitulo4.js
// Requiere Chromium en /usr/bin/chromium y el servicio desplegado.
const puppeteer = require('puppeteer-core');
const URL = 'https://salarios.debugsito.click';
const SALIDA = __dirname + '/../figuras/servicio';

async function consultar(p, pais, rol, anios) {
  await p.goto(URL, {waitUntil: 'networkidle0'});
  await new Promise(r => setTimeout(r, 1500));
  await p.evaluate(async (pais, rol, anios) => {
    document.querySelector('#pais').value = pais;
    await cargarRoles();
    document.querySelector('#rol').value = rol;
    document.querySelector('#anios_profesionales').value = anios;
    document.querySelector('#enviar').click();
  }, pais, rol, anios);
  await new Promise(r => setTimeout(r, 4000));
}

(async () => {
  const b = await puppeteer.launch({executablePath: '/usr/bin/chromium',
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
    defaultViewport: {width: 1180, height: 900, deviceScaleFactor: 2}});
  const p = await b.newPage();

  // La abstención: Perú no recibe cifra, y la interfaz explica por qué.
  await consultar(p, 'Peru', 'Developer, back-end', 8);
  await p.screenshot({path: `${SALIDA}/peru_rechazo.png`});
  console.log('peru_rechazo capturada');

  // La traza técnica: latencia y respuesta JSON desplegada.
  await consultar(p, 'Germany', 'Developer, back-end', 8);
  await p.evaluate(() => {
    const d = document.querySelector('details.json');
    if (d) { d.open = true; d.scrollIntoView({block: 'center'}); }
  });
  await new Promise(r => setTimeout(r, 500));
  await p.screenshot({path: `${SALIDA}/json_abierto.png`});
  console.log('json_abierto capturada');

  // El «acerca de».
  await p.goto(URL, {waitUntil: 'networkidle0'});
  await new Promise(r => setTimeout(r, 1200));
  await p.evaluate(() => abrirAcerca());
  await new Promise(r => setTimeout(r, 600));
  await p.screenshot({path: `${SALIDA}/acerca.png`});
  console.log('acerca capturada');

  // La documentación interactiva que FastAPI genera del contrato.
  await p.goto(`${URL}/api/docs`, {waitUntil: 'networkidle0'});
  await new Promise(r => setTimeout(r, 1500));
  await p.screenshot({path: `${SALIDA}/api_docs.png`});
  console.log('api_docs capturada');

  await b.close();
})();
