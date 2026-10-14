// Capturas para el capítulo de implementación: la abstención sobre Perú, el
// JSON de respuesta, el «acerca de» y la documentación de la API.
// Mismos requisitos que capturas_anexo_k.js.
//   node scripts/capturas_capitulo4.js
const puppeteer = require('puppeteer-core');
const URL = 'https://salarios.debugsito.click';
const SALIDA = __dirname + '/../figuras/servicio';

// Recorta hasta el final de los paneles: sin el pie de página ni el hueco de debajo.
async function hastaPaneles(p) {
  const fin = await p.evaluate(() =>
    document.querySelector('#formulario').parentElement.getBoundingClientRect().bottom + window.scrollY);
  return {x: 0, y: 0, width: 900, height: Math.ceil(fin) + 16};
}

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
    defaultViewport: {width: 900, height: 1200, deviceScaleFactor: 2}});
  const p = await b.newPage();

  // Perú no recibe cifra; la interfaz explica por qué.
  await consultar(p, 'Peru', 'Developer, back-end', 8);
  await p.screenshot({path: `${SALIDA}/peru_rechazo.png`, clip: await hastaPaneles(p)});
  console.log('peru_rechazo capturada');

  // Respuesta JSON desplegada.
  await consultar(p, 'Germany', 'Developer, back-end', 8);
  // La respuesta íntegra no cabe en la columna de resultados: se oculta el formulario
  // y el JSON se reparte en dos columnas a todo el ancho, con letra legible impresa.
  const zona = await p.evaluate(() => {
    document.querySelector('#formulario').style.display = 'none';
    const cont = document.querySelector('#formulario').parentElement;
    cont.style.display = 'block';
    const d = document.querySelector('details.json');
    d.open = true;
    d.style.fontSize = '15px';
    Object.assign(d.querySelector('pre').style, {maxHeight: 'none', overflow: 'visible',
      whiteSpace: 'pre-wrap', fontSize: '15px', columnCount: 2, columnGap: '28px'});
    const ini = document.querySelector('p.latencia').getBoundingClientRect();
    const fin = d.getBoundingClientRect();
    return {x: Math.floor(fin.left) - 12, y: Math.floor(ini.top + scrollY) - 12,
            width: Math.ceil(fin.width) + 24, height: Math.ceil(fin.bottom - ini.top) + 24};
  });
  await new Promise(r => setTimeout(r, 400));
  await p.screenshot({path: `${SALIDA}/json_abierto.png`, clip: zona});
  console.log('json_abierto capturada');

  await p.goto(URL, {waitUntil: 'networkidle0'});
  await new Promise(r => setTimeout(r, 1200));
  await p.evaluate(() => { abrirAcerca(); document.activeElement.blur(); });
  await new Promise(r => setTimeout(r, 600));
  await (await p.$('dialog#acerca')).screenshot({path: `${SALIDA}/acerca.png`});
  console.log('acerca capturada');

  // Documentación que FastAPI genera sola.
  await p.goto(`${URL}/api/docs`, {waitUntil: 'networkidle0'});
  await new Promise(r => setTimeout(r, 1500));
  await p.screenshot({path: `${SALIDA}/api_docs.png`, clip: {x: 0, y: 0, width: 900, height: 900}});
  console.log('api_docs capturada');

  await b.close();
})();
