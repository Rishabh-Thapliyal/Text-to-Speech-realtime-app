// Minimal Node script to convert TeX to spoken text using MathJax + Speech Rule Engine (SRE)
// Usage: node math_speech.js "\\frac{a}{b}" clearspeak

async function main() {
  const args = process.argv.slice(2);
  const tex = args[0] || '';
  const style = (args[1] || 'clearspeak').toLowerCase();

  try {
    const sre = require('speech-rule-engine');
    const { mathjax } = require('mathjax-full/js/mathjax.js');
    const { TeX } = require('mathjax-full/js/input/tex.js');
    const { liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js');
    const { RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js');
    const { MathML } = require('mathjax-full/js/input/mathml.js');

    // Initialize SRE
    sre.setupEngine({
      domain: style === 'mathspeak' ? 'mathspeak' : 'clearspeak',
      style: 'default',
      speech: 'shallow',
      semantics: true,
      locale: 'en'
    });

    const adaptor = liteAdaptor();
    RegisterHTMLHandler(adaptor);

    const texInput = new TeX({packages: ['base', 'ams']});
    const mmlInput = new MathML();

    const html = mathjax.document('', {InputJax: texInput});

    // Convert TeX to internal MathJax node, then to serialized MathML
    const node = html.convert(tex, {display: false});
    const mmlSerialized = adaptor.outerHTML(node);

    // Use SRE to convert MathML to speech
    const speech = sre.toSpeech(mmlSerialized);

    process.stdout.write((speech || '').trim());
  } catch (err) {
    // Fallback: return the TeX with minimal cleanup, no backslashes spoken
    process.stdout.write(tex.replace(/\\/g, ' ').replace(/[{}]/g, ' ').trim());
  }
}

main();
