# Strata Private — fork de Strata con modo low-RAM sin `experts.bin`

Este repositorio es un **fork** de [Strata](https://github.com/Niko1221/Strata).

Aquí solo se documenta **lo que este fork añade** sobre el repositorio original. Para instalar, configurar
o entender Strata en sí (modelos, motores, la app, la Terminal, las APIs), ve al repositorio original:
**[github.com/Niko1221/Strata](https://github.com/Niko1221/Strata)**.

- Repositorio original (upstream): <https://github.com/Niko1221/Strata>
- Este fork: <https://github.com/JuanSystems/Strata-Private>

---

## Qué añade este fork

Tres cosas. Dos giran alrededor del mismo problema — **el archivo `experts.bin` ocupa entre 35 y 50 GB
que quizá no tienes** — y la tercera es una función que el motor original no trae: **guardar las
conversaciones en disco para que un reinicio o una expulsión de la caché no las deseche.**

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

### 3. Conversaciones que sobreviven al disco: `--conversation-spill-gib`

El motor de Strata guarda en RAM las conversaciones en pausa (las "parked") para que el siguiente turno
empiece en segundos. Cuando esa RAM se llena, la conversación más antigua se descartaba: el turno
siguiente volvía a leer el prompt entero (unos 210 s para 72K tokens). Este fork añade un **segundo
nivel en disco**: la conversación expulsada se escribe a un archivo y el turno siguiente la restaura en
1-2 s. Nada se pierde mientras quede disco.

Se activa con un argumento del motor, apagado por defecto:

```sh
--conversation-spill-gib 64
```

| Argumento | Qué hace | Por defecto |
|-----------|----------|-------------|
| `--conversation-spill-gib N` | Presupuesto en GiB del nivel de disco. `0` lo apaga y el motor se comporta exactamente como antes | `0` (apagado) |
| `--conversation-spill-path PATH` | Carpeta de los archivos de conversación | `<data_dir>/strata-spill/` |
| `--conversation-spill-min-tokens N` | Solo se guardan conversaciones de al menos N tokens | hereda `--conversation-cache-min-tokens` |
| `--conversation-idle-spill-seconds N` | Tras el `DONE`, espera N s a otro turno; si no llega, respalda la conversación viva al disco. `0` respalda al instante; negativo lo apaga | `5` |

Lo que conviene saber antes de activarlo:

- **Sobrevive a reinicios.** Al arrancar, el motor relee el manifiesto del disco; una conversación que
guardó la ejecución anterior se puede restaurar sin releer el prompt. Los archivos de una ejecución
interrumpida a medias se limpian solos.
- **El presupuesto es un LRU.** Cuando el nivel de disco se llena, se borra la conversación usada hace
más tiempo, no la más nueva.
- **Un prefijo compartido (`pin=N`) también se guarda.** Su instantánea `pfx-*.bin` queda en el mismo
directorio, así que el caché entre conversaciones tampoco depende de que el prefijo se acabe de crear en
el request actual.
- **Si el disco está lleno o el archivo está corrupto, no rompe nada.** La conversación se descarta y el
motor cae en el comportamiento de siempre: releer el prompt.
- **A/B seguro.** Con `--conversation-spill-gib 0` no se crea ningún archivo y el camino es el del motor
original, byte por byte.

En la configuración `strata-<model>.json` de la app se escribe como un bloque, y el servidor lo traduce a
los argumentos de arriba:

```json
"conversation_spill": { "budget_gib": 64, "path": "D:/strata-spill", "min_tokens": 2000, "idle_seconds": 5 }
```

Cada GiB de presupuesto es espacio en disco, no en RAM: una conversación de 72K tokens ocupa cerca de
1.4 GiB de archivo. Un valor entre 32 y 128 GiB cubre un día de trabajo normal. Esta función es propia de
este fork; la versión original de Strata no la tiene.

---

## Compilar el motor desde el código fuente (Linux + CUDA)

El motor se puede compilar a mano cuando quieres una arquitectura de CUDA concreta, una build para tu
tarjeta exacta o una versión que el instalador no trae. Los dos ajustes que más importan:

### Release, no Debug (esto es lo que más pesa)

**Compila siempre con `-DCMAKE_BUILD_TYPE=Release`.** Una build `Debug` (o una build sin tipo, que es lo
que CMake hace si no le dices nada) apaga las optimizaciones del compilador: los kernels de decodificación
dejan de usar los registros y las instrucciones vectorizadas que asumen, y el motor pasa de unos **30
tokens/s a dificilmente 15 tokens/s** en una tarjeta que antes iba bien. El síntoma engaña porque la build
funciona y da respuestas correctas — solo va a la mitad. Si compilaste una vez en `Debug` y no entiendes
la caída de velocidad, no busques en los parámetros del modelo ni en el `--conversation-cache-mib`: vuelve
a configurar el proyecto en `Release` con un directorio de build limpio.

```sh
rm -rf build   # una build Debug NO se convierte en Release cambiando una opción: hay que empezar de cero
```

### La arquitectura de CUDA es la de TU tarjeta

`-DCMAKE_CUDA_ARCHITECTURES` no es un valor fijo: es el *compute capability* de la tarjeta donde va a
correr el motor. Si pones un número que no es el tuyo, el motor arranca pero no aprovecha los kernels de
la generación correcta (o directamente no arranca).

| Tarjeta | Valor |
|---------|-------|
| RTX 20xx (Turing) | `75` |
| RTX 30xx (Ampere) | `86` |
| RTX 40xx (Ada) | `89` |
| RTX 50xx (Blackwell) | `120` |
| Sin especificar | [`CMakeLists.txt`](CMakeLists.txt:194) usa `120` |

Puedes poner varias separadas por punto y coma (`-DCMAKE_CUDA_ARCHITECTURES="75;86;89"`) si el binario
va a circular entre varias máquinas, pero cada arquitectura extra alarga la compilación. El proyecto
rechaza cualquier valor por debajo de `75` salvo que compiles con `-DSTRATA_EXPERIMENTAL_SM60=ON`
([`CMakeLists.txt`](CMakeLists.txt:193), solo CUDA 12.x).

### El comando completo

Necesitas CMake 3.24 o más nuevo, Ninja, un compilador C++17 y el *toolkit* de CUDA instalado. El comando
busca el toolkit con `nvcc` en el `PATH`, así que no lleva rutas de una versión concreta de CUDA: funciona
con la que tengas (12.x o 13.x) sin editarlo.

```sh
# 1. Localiza el toolkit que tienes instalado (sirve para 12.x y 13.x)
CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(which nvcc)")")")"
echo "CUDA toolkit: $CUDA_HOME"   # p. ej. /usr/local/cuda-13.0

# 2. Configura
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/nvidia/lib64 -L$CUDA_HOME/targets/x86_64-linux/lib -Wl,--no-as-needed -lcuda -lcublas_static -lcublasLt_static -lculibos -lpthread -ldl -lrt -Wl,--as-needed -static-libgcc -static-libstdc++" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/nvidia/lib64 -L$CUDA_HOME/targets/x86_64-linux/lib -Wl,--no-as-needed -lcuda -lcublas_static -lcublasLt_static -lculibos -lpthread -ldl -lrt -Wl,--as-needed -static-libgcc -static-libstdc++" \
  -DCMAKE_MODULE_LINKER_FLAGS="-L/usr/local/nvidia/lib64 -L$CUDA_HOME/targets/x86_64-linux/lib -Wl,--no-as-needed -lcuda -lcublas_static -lcublasLt_static -lculibos -lpthread -ldl -lrt -Wl,--as-needed -static-libgcc -static-libstdc++"

# 3. Compila el motor
cmake --build build --target strata -j2
```

Cambia `-DCMAKE_CUDA_ARCHITECTURES=75` por el valor de tu tarjeta de la tabla de arriba. Sube `-j2` si
tienes RAM suficiente para más procesos de `nvcc` a la vez (cada uno puede consumir varios GB).

Por qué los tres bloques de `*_LINKER_FLAGS` llevan las rutas del toolkit: enlazan `libcublas`,
`libcublasLt` y `libcuda` de forma **estática** (`-lcublas_static`, `-lcublasLt_static`). El binario
resultante no depende de las `.so` del toolkit en tiempo de ejecución, así que funciona en un contenedor o
en una máquina con otro CUDA instalado. Si no necesitas eso, puedes omitir los tres bloques y dejar que
CMake enlace con los *imported targets* de `CUDA::cublas` y `CUDA::cudart` que ya usa
([`CMakeLists.txt`](CMakeLists.txt:191)).

`-DCMAKE_C_COMPILER_LAUNCHER=ccache` (y sus dos hermanos) son opcionales, pero con `ccache` instalado
una recompilación después de tocar un solo archivo pasa de minutos a segundos.

### Comprobar que quedó en Release

```sh
# Debe decir CMAKE_BUILD_TYPE:STRING=Release (vacío o Debug = vuelve al paso 1)
grep CMAKE_BUILD_TYPE build/CMakeCache.txt
```

Si `CMAKE_BUILD_TYPE` sale vacío o `Debug`, la velocidad de generación se queda a la mitad aunque todo
compile y responda bien.

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
