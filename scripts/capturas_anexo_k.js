// Capturas del prototipo para el Anexo K del documento.
//
// Uso:  npm install puppeteer-core   (donde sea; pasar NODE_PATH)
//       node scripts/capturas_anexo_k.js
// Requiere Chromium en /usr/bin/chromium y el servicio desplegado.
const puppeteer = require('puppeteer-core');
const URL = 'https://salarios.debugsito.click';
const SALIDA = __dirname + '/../figures/servicio';

(async () => {
  const b = await puppeteer.launch({executablePath: '/usr/bin/chromium',
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
    defaultViewport: {width: 1180, height: 900, deviceScaleFactor: 2}});
  const p = await b.newPage();
  const casos = [
    ['candidato', 'candidato', 'Germany', 'Developer, back-end', 8, [], false],
    ['reclutador', 'reclutador', 'Brazil', 'Developer, full-stack', 6, [], false],
    ['comparar', 'comparar', 'Peru', 'Developer, back-end', 8,
      ['United States of America', 'Germany', 'Brazil', 'India'], false],
    ['comparar_ppa', 'comparar', 'Peru', 'Developer, back-end', 8,
      ['United States of America', 'Germany', 'Brazil', 'India'], true],
  ];
  for (const [nombre, vista, pais, rol, anios, comparar, ppa] of casos) {
    await p.goto(URL, {waitUntil: 'networkidle0'});
    await new Promise(r => setTimeout(r, 1500));
    await p.evaluate(async (vista, pais, rol, anios, comparar) => {
      document.querySelector(`[data-vista=${vista}]`).click();
      document.querySelector('#pais').value = pais;
      await cargarRoles();
      document.querySelector('#rol').value = rol;
      document.querySelector('#anios_profesionales').value = anios;
      if (comparar.length) {
        const s = document.querySelector('#comparar_paises');
        for (const c of comparar) {
          const o = [...s.options].find(x => x.value === c);
          if (o) o.selected = true;
        }
      }
      document.querySelector('#enviar').click();
    }, vista, pais, rol, anios, comparar);
    await new Promise(r => setTimeout(r, 4000));
    if (ppa) {
      await p.evaluate(() => document.querySelector('#escala-ppa').click());
      await new Promise(r => setTimeout(r, 800));
    }
    await p.screenshot({path: `${SALIDA}/${nombre}.png`});
    console.log(nombre, 'capturada');
  }
  await b.close();
})();
