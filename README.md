# Strata Private — fork de Strata con modo low-RAM sin `experts.bin`

Este repositorio es un **fork** de [Strata](https://github.com/Niko1221/Strata).

Aquí solo se documenta **lo que este fork añade** sobre el repositorio original. Para instalar, configurar
o entender Strata en sí (modelos, motores, la app, la Terminal, las APIs), ve al repositorio original:
**[github.com/Niko1221/Strata](https://github.com/Niko1221/Strata)**.

- Repositorio original (upstream): <https://github.com/Niko1221/Strata>
- Este fork: <https://github.com/JuanSystems/Strata-Private>

---

## Qué añade este fork

Dos cosas, las dos alrededor del mismo problema: **el archivo `experts.bin` ocupa entre 35 y 50 GB que
quizá no tienes.**

### 1. Modo low-RAM que no escribe `experts.bin`

El motor de Strata ya sabía leer los expertos directamente de los shards GGUF cuando el pack no tiene
`experts.bin` (desde el motor 0.1.31). Lo que faltaba era que el setup dejara de escribirlo. Ahora lo hace
con una variable de entorno:

```sh
STRATA_NO_EXPERTS_BIN=1
```

Con ella, el setup descarga los GGUFs y **no crea** el archivo de expertos. El espacio ahorrado es la
copia que el modo low-RAM hacía antes: unos 35 GB en IQ2_XS y unos 50 GB en IQ3_S.

Si quieres forzar la variante que precarga en RAM los expertos que la GPU no retiene, añade también:

```sh
STRATA_RESIDENT_VARIANT=1
```

Es solo un escape hatch: en la mayoría de los PCs el comportamiento automático (page cache del sistema
operativo) rinde mejor.

Sin `STRATA_NO_EXPERTS_BIN`, el setup escribe `experts.bin` exactamente como antes. **El camino por
defecto no cambia ni un byte.**

### 2. `tools/strata_extract_experts_bin.py` — construir el pack perforando el GGUF

#### Por qué se perfora el GGUF

Hay dos formas de darle los expertos al motor y cada una gana en algo distinto:

- **Leerlos del GGUF en sitio** ahorra disco, pero la inferencia es más lenta: cada experto son tres
  lecturas separadas (gate, up, down) esparcidas por el archivo, en vez de un bloque contiguo.
- **Tenerlos en `experts.bin`** es más rápido, porque cada experto es un bloque contiguo que el motor
  lee de una vez, pero eso **duplica** los expertos: siguen dentro del GGUF *y* están otra vez en el
  archivo nuevo. De ahí los ~105 GB.

Perforar el GGUF resuelve la duplicación. Cuando los expertos de un rango ya se copiaron a `experts.bin`,
ese rango del GGUF se convierte en un hueco: el archivo sigue midiendo lo mismo de cara al sistema
(las posiciones no cambian, `native_experts.txt` sigue siendo válido), pero esos bytes **dejan de ocupar
disco**. El GGUF queda conteniendo solo lo que el modelo sigue leyendo de él —las partes densas, los
routers, las tablas—, y los expertos se leen del `experts.bin` contiguo.

El resultado es lo mejor de los dos mundos: **la velocidad de inferencia de `experts.bin` sin pagar dos
veces el espacio de los expertos.**

#### Uso

Para discos muy ajustados. El setup estándar necesita los GGUFs (~70 GB) **más** `experts.bin`
(~35 GB) al mismo tiempo: unos 105 GB. Si tienes 90 GB libres y ya bajaste los GGUFs, el setup falla.

Esta herramienta extrae los expertos capa por capa a un archivo intermedio, **perfora** (hole-punch)
los rangos del GGUF de los que ya se copiaron y luego concatena. El uso de disco se mantiene alrededor
de 70 GB durante toda la operación, en vez de crecer a 105 GB.

```sh
python3 tools/strata_extract_experts_bin.py \
    --gguf /ruta/al/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
    --out  /ruta/al/data/packs/iq2xs \
    --phase all
```

Fases, por si prefieres ir paso a paso y verificar entre medias:

| Fase | Qué hace |
|------|----------|
| `--phase 1` | extrae los expertos del shard 1 a `expert-01.bin` y perfora el shard 1 |
| `--phase 2` | extrae los expertos del shard 2 a `expert-02.bin` y perfora el shard 2 |
| `--phase 3` | concatena los intermedios en `experts.bin` (streaming, sin pico de disco) |
| `--phase 4` | escribe el sidecar `experts.bin.src.json` y actualiza las marcas `.done` |
| `--phase all` | las cuatro, en orden |

Si una fase se interrumpe, las fases 1 y 2 retoman desde las capas registradas en
`expert-NN.bin.progress.json`. No empiezan de cero.

**La herramienta comprueba antes de tocar nada que tu sistema de archivos soporte hole-punch.** Si no
lo soporta, aborta con un mensaje en vez de dejarte creyendo que ahorraste 35 GB que en realidad no se
liberaron.

---

## Cómo subir tu GGUF perforado a Hugging Face

Un GGUF perforado **no se puede subir tal cual**. Hugging Face no guarda archivos sparse: cuando subes un
archivo de 70 GB, lo rellena y vuelve a ocupar los 70 GB reales, perdiendo todo el ahorro. Por eso hay que
empaquetarlo primero con `tar --sparse` (que conserva los huecos) y comprimirlo (los ceros se comprimen casi
a nada).

### Paso 1: empaquetar con `tar --sparse` + compresión

```bash
# Empaquetar (preserva sparse + comprime los ceros restantes muy bien)
tar --sparse -cf - -C /tmp/strata-data/models/IQ2_XS \
    Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  | gzip -1 > /kaggle/working/Qwen-IQ2_XS-shard1-sparse.tar.gz
```

Ajusta la ruta `-C` al directorio donde están tus GGUFs y el nombre del `.gguf` al de tu shard. Repite el
comando para el shard 2 si también lo perforaste.

`-1` es el nivel de compresión más rápido. Los huecos son ceros y comprimen igual de bien en cualquier
nivel, así que no hace falta subir a `-9`: tardarías mucho más para ganar muy poco.

### Paso 2: subir el `.tar.gz`

Sube el archivo resultante a tu repositorio de Hugging Face con la interfaz web o con `huggingface-cli`:

```bash
huggingface-cli upload <usuario>/<repositorio> /kaggle/working/Qwen-IQ2_XS-shard1-sparse.tar.gz
```

### Paso 3: al descargarlo, descomprimir conservando los huecos

```bash
tar --sparse -xzf Qwen-IQ2_XS-shard1-sparse.tar.gz -C /tmp/strata-data/models/IQ2_XS
```

`--sparse` en la extracción es imprescindible: sin él, `tar` escribe los ceros como datos reales y el
archivo vuelve a ocupar los 70 GB.

### Verificar que el ahorro sobrevivió

```bash
# "aparente" vs "real en disco": deben ser distintos
ls -lh  Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
du -h   Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
```

Si `du` reporta bastante menos que `ls`, los huecos siguen ahí.

### Recordatorio

Las marcas `.done` de los GGUFs quedan con la línea `trimmed:`, así que el setup no intentará rehashear
111 GB de ceros ni volverá a descargar los archivos. Si compartes el pack, incluye también el
`experts.bin.src.json` que dejó la fase 4: el motor lo usa para saber a qué modelo pertenece cada
`experts.bin`.

---

## Tests

```sh
python tools/test_setup_lowram.py
python tools/test_extract_experts_bin.py
```

Los dos son `unittest`, sin GPU y sin descargas.

---

## Créditos y licencia

Todo el trabajo del motor, el setup, la app y la Terminal es de **[Strata](https://github.com/Niko1221/Strata)**
y sus contribuidores. Este fork solo añade lo descrito arriba. La licencia es la misma del proyecto original
(ver [LICENSE](LICENSE)).
