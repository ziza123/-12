/**
 * useSignModel — React hook for loading and running the Qyran sign language model.
 *
 * Usage:
 *   const { predict, isLoaded, labels, featuresPerFrame } = useSignModel();
 *
 *   // landmarks = Float32Array of shape [60, featuresPerFrame] flattened,
 *   // or number[][] of shape [60][featuresPerFrame]
 *   const result = await predict(landmarks);
 *   // result = { label: "хороший", confidence: 0.92, top5: [...], probs: Float32Array }
 *
 * Supports both the legacy model ([1,60,255], StandardScaler in scaler.json)
 * and the new model ([1,60,259], no scaler, includes a 'no_event' class).
 * The per-frame feature count is derived from the loaded model's input shape.
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';

const DEBUG = false;

interface PredictionResult {
  label: string;
  confidence: number;
  classIndex: number;
  top5: { label: string; confidence: number }[];
  /** Full probability vector over all classes (softmax output). */
  probs: Float32Array;
}

interface UseSignModelReturn {
  predict: (landmarks: Float32Array | number[][]) => Promise<PredictionResult | null>;
  isLoaded: boolean;
  isLoading: boolean;
  error: string | null;
  labels: string[];
  numClasses: number;
  /** Per-frame feature count derived from the model input shape (255 legacy, 259 new). */
  featuresPerFrame: number;
  /**
   * Download progress of the weights, 0..1. The model is ~8 MB, which is tens
   * of seconds on a weak connection — without a number the page just sits on
   * "Loading" and reads as frozen.
   */
  progress: number;
  labelMap: Record<string, number>;
  idxToLabel: Record<number, string>;
  /** Сколько знаков разрешено сценарием; null — ограничения нет. */
  vocabSize: number | null;
}

const MODEL_URL = '/model/model.json';
const LABEL_MAP_URL = '/model/label_map.json';
const SCALER_URL = '/model/scaler.json';
const VOCAB_URL = '/model/vocabularies.json';
const SEQ_LEN = 60;
const DEFAULT_FEATURES = 255;

interface UseSignModelOptions {
  /**
   * Ограничить выход словарём сценария из vocabularies.json.
   *
   * Модель обучена на 1001 знак, но за конкретной стойкой показывают две
   * сотни. Зануление недопустимых классов перед argmax поднимает точность на
   * невиданных подписантах с 58.5% до 80.9% — измерено, ml/vocab_experiment.py.
   * Дообучение на том же словаре дало +0.2 пункта при стандартной ошибке 1.8,
   * то есть ничего: решает сужение словаря, а не способ его добиться. Поэтому
   * одна модель и словарь на площадку, а не модель на площадку.
   *
   * undefined — без ограничения, все 1001 класс.
   */
  vocabulary?: string;
}

export function useSignModel(options: UseSignModelOptions = {}): UseSignModelReturn {
  const { vocabulary } = options;
  /** Индексы классов, разрешённых сценарием. null — ограничения нет. */
  const allowedRef = useRef<Uint8Array | null>(null);
  const [vocabSize, setVocabSize] = useState<number | null>(null);
  const modelRef = useRef<tf.GraphModel | tf.LayersModel | null>(null);
  // Pre-built scaler tensors (built ONCE at load time, null when no scaler.json).
  const scalerMeanRef = useRef<tf.Tensor1D | null>(null);
  const scalerScaleRef = useRef<tf.Tensor1D | null>(null);
  const featuresRef = useRef<number>(DEFAULT_FEATURES);
  const [featuresPerFrame, setFeaturesPerFrame] = useState<number>(DEFAULT_FEATURES);
  const [labelMap, setLabelMap] = useState<Record<string, number>>({});
  const [idxToLabel, setIdxToLabel] = useState<Record<number, string>>({});
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Load model + label map + optional scaler
  useEffect(() => {
    let cancelled = false;

    async function load() {
      setIsLoading(true);
      setError(null);

      try {
        // Load label map and scaler in parallel. scaler.json is OPTIONAL:
        // the new model ships without one (404 / network error -> null).
        const [lmRes, scalerRes] = await Promise.all([
          fetch(LABEL_MAP_URL),
          fetch(SCALER_URL).catch(() => null),
        ]);

        if (!lmRes.ok) throw new Error(`Failed to load label_map.json: ${lmRes.status}`);

        let scalerData: { mean: number[]; scale: number[] } | null = null;
        if (scalerRes && scalerRes.ok) {
          try {
            scalerData = await scalerRes.json();
          } catch {
            scalerData = null;
          }
        }

        const lm: Record<string, number> = await lmRes.json();
        if (cancelled) return;

        setLabelMap(lm);
        const i2l: Record<number, string> = {};
        for (const [label, idx] of Object.entries(lm)) {
          i2l[idx] = label;
        }
        setIdxToLabel(i2l);

        // Словарь сценария. Отсутствие файла или неизвестное имя — не ошибка:
        // распознаватель просто работает на полном словаре, как раньше.
        allowedRef.current = null;
        setVocabSize(null);
        if (vocabulary) {
          try {
            const vRes = await fetch(VOCAB_URL);
            const all: Record<string, { words: string[] }> = await vRes.json();
            const words = all[vocabulary]?.words;
            if (!words) throw new Error(`нет словаря «${vocabulary}»`);
            const mask = new Uint8Array(Object.keys(lm).length);
            let n = 0;
            for (const w of words) {
              const i = lm[w];
              if (i !== undefined && !mask[i]) { mask[i] = 1; n++; }
            }
            allowedRef.current = mask;
            setVocabSize(n);
          } catch (e) {
            console.warn('[Qyran] словарь сценария не применён:', e);
          }
        }

        // Load as layers model (Keras export)
        // Прогресс загрузки.
        //
        // Родной onProgress у tf.js щёлкает раз на ФАЙЛ, а весов всего два
        // шарда — получается 0 / 50 / 100 и долгое молчание между. Поэтому
        // считаем байты сами: подсовываем свой fetch, читаем тело потоком и
        // отдаём дальше те же байты. Лишних запросов это не добавляет.
        // Считаем только шарды весов (.bin) — на них приходятся все 8 МБ.
        // Мешать в счётчик model.json нельзя: он приходит первым, и прогресс
        // сразу показал бы 100%, а потом висел бы на нём всю настоящую загрузку.
        let loaded = 0;
        let total = 0;
        const onProgress = (fraction: number) => {
          // Запасной путь: если байты посчитать не вышло, берём грубый
          // пофайловый прогресс от tf.js.
          if (!cancelled && total === 0) setProgress(Math.max(0, Math.min(1, fraction)));
        };

        const fetchFunc: typeof fetch = async (input, init) => {
          const url = typeof input === 'string' ? input : (input as Request).url ?? '';
          const res = await fetch(input as RequestInfo, init);
          const len = Number(res.headers.get('content-length') || 0);
          // Без content-length (сжатие «на лету», chunked) считать нечего.
          if (!/\.bin(\?|$)/.test(url) || !res.body || !len || !res.ok) return res;

          total += len;
          const reader = res.body.getReader();
          const chunks: Uint8Array[] = [];
          for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            loaded += value.length;
            if (!cancelled) setProgress(Math.min(1, loaded / total));
          }
          return new Response(new Blob(chunks as BlobPart[]), {
            status: res.status,
            statusText: res.statusText,
            headers: res.headers,
          });
        };

        let model: tf.GraphModel | tf.LayersModel;
        try {
          model = await tf.loadLayersModel(MODEL_URL, { strict: false, onProgress, fetchFunc });
          console.log('[Qyran] Loaded as LayersModel');
        } catch (e1: any) {
          console.warn('[Qyran] LayersModel failed:', e1.message);
          try {
            model = await tf.loadGraphModel(MODEL_URL, { onProgress, fetchFunc });
            console.log('[Qyran] Loaded as GraphModel');
          } catch (e2: any) {
            throw new Error(`LayersModel: ${e1.message} | GraphModel: ${e2.message}`);
          }
        }

        if (cancelled) { model.dispose(); return; }

        // Derive per-frame feature count from the model's input shape [null, 60, F]
        let features = DEFAULT_FEATURES;
        try {
          const inShape = model.inputs?.[0]?.shape;
          const last = inShape ? inShape[inShape.length - 1] : null;
          if (typeof last === 'number' && last > 0) features = last;
        } catch {
          // keep default
        }
        featuresRef.current = features;
        setFeaturesPerFrame(features);

        // Pre-build scaler tensors ONCE (avoids per-predict tensor allocation).
        // Skip entirely when no scaler.json (new model needs no scaling) or on
        // a dimension mismatch (e.g. stale scaler next to a new model).
        if (scalerData && Array.isArray(scalerData.mean) && Array.isArray(scalerData.scale)) {
          if (scalerData.mean.length === features && scalerData.scale.length === features) {
            scalerMeanRef.current = tf.tensor1d(scalerData.mean);
            scalerScaleRef.current = tf.tensor1d(scalerData.scale);
            console.log('[Qyran] Scaler loaded');
          } else {
            console.warn('[Qyran] scaler.json dims do not match model input — skipping scaling');
          }
        } else {
          console.log('[Qyran] No scaler.json — running without input scaling');
        }

        modelRef.current = model;
        setIsLoaded(true);
        console.log(`[Qyran] Model ready. Classes: ${Object.keys(lm).length}, features/frame: ${features}`);

        // Warmup inference
        const dummy = tf.zeros([1, SEQ_LEN, features]);
        const warmup = model instanceof tf.GraphModel
          ? model.predict(dummy)
          : (model as tf.LayersModel).predict(dummy);
        if (warmup instanceof tf.Tensor) warmup.dispose();
        else if (Array.isArray(warmup)) warmup.forEach(t => t.dispose());
        dummy.dispose();
        console.log('[Qyran] Warmup done');
      } catch (err: any) {
        if (!cancelled) {
          console.error('[Qyran] Load error:', err);
          setError(err.message || 'Failed to load model');
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    load();
    return () => { cancelled = true; };
  }, [vocabulary]);

  // Predict
  const predict = useCallback(async (
    landmarks: Float32Array | number[][]
  ): Promise<PredictionResult | null> => {
    const model = modelRef.current;
    if (!model || Object.keys(idxToLabel).length === 0) return null;

    const features = featuresRef.current;

    return tf.tidy(() => {
      // Convert input to tensor [1, 60, features]
      let input: tf.Tensor3D;

      if (landmarks instanceof Float32Array) {
        if (landmarks.length !== SEQ_LEN * features) {
          console.warn(`[Qyran] Expected ${SEQ_LEN * features} values, got ${landmarks.length}`);
          return null;
        }
        input = tf.tensor3d(landmarks, [1, SEQ_LEN, features]);
      } else {
        if (landmarks.length !== SEQ_LEN || (landmarks[0] && landmarks[0].length !== features)) {
          console.warn(`[Qyran] Expected [${SEQ_LEN}][${features}] frames, got [${landmarks.length}][${landmarks[0]?.length}]`);
          return null;
        }
        input = tf.tensor3d([landmarks], [1, SEQ_LEN, features]);
      }

      // Apply scaler normalization: (x - mean) / scale (legacy model only).
      // Tensors are pre-built at load time; tf.tidy does not dispose them
      // because they were created outside this scope.
      const mean = scalerMeanRef.current;
      const scale = scalerScaleRef.current;
      if (mean && scale) {
        input = input.sub(mean).div(scale) as tf.Tensor3D;
      }

      // Run inference
      const output = model instanceof tf.GraphModel
        ? model.predict(input)
        : (model as tf.LayersModel).predict(input);

      const probsTensor = output instanceof tf.Tensor ? output : (output as tf.Tensor[])[0];
      const raw = probsTensor.dataSync() as Float32Array;

      // Ограничение словарём применяется К САМОМУ ВЕКТОРУ probs, а не только к
      // top5. Потребители берут именно probs и считают свой argmax поверх
      // сглаживания (RecognizerPage ведёт EMA по всему вектору) — фильтрация
      // только на выходе top5 не влияла бы ни на одно решение.
      //
      // Массу не перенормируем: уверенность должна остаться в той же шкале,
      // на которой откалиброван порог тишины.
      const allowed = allowedRef.current;
      const probs = allowed ? Float32Array.from(raw) : raw;
      if (allowed) {
        for (let i = 0; i < probs.length; i++) if (allowed[i] !== 1) probs[i] = 0;
      }

      const indexed = Array.from(probs)
        .map((p, i) => ({ idx: i, prob: p }))
        .filter(({ idx }) => !allowed || allowed[idx] === 1);
      indexed.sort((a, b) => b.prob - a.prob);
      if (indexed.length === 0) return null;
      const top5 = indexed.slice(0, 5).map(({ idx, prob }) => ({
        label: idxToLabel[idx] || `class_${idx}`,
        confidence: prob,
      }));

      // Debug logging
      if (DEBUG) {
        const probArr = Array.from(probs);
        const maxProb = Math.max(...probArr);
        const minProb = Math.min(...probArr);
        const sumProb = probArr.reduce((a, b) => a + b, 0);
        console.log(
          '[Qyran] Predict →',
          top5[0].label,
          `(${(top5[0].confidence * 100).toFixed(1)}%)`,
          '| range:', minProb.toFixed(4), '-', maxProb.toFixed(4),
          '| sum:', sumProb.toFixed(3),
          '| top5:', top5.map(t => `${t.label}=${(t.confidence*100).toFixed(0)}%`).join(', ')
        );
      }

      return {
        label: top5[0].label,
        confidence: top5[0].confidence,
        classIndex: indexed[0].idx,
        top5,
        probs,
      };
    });
  }, [idxToLabel]);

  return {
    predict,
    isLoaded,
    isLoading,
    error,
    labels: Object.keys(labelMap),
    numClasses: Object.keys(labelMap).length,
    featuresPerFrame,
    progress,
    labelMap,
    idxToLabel,
    vocabSize,
  };
}
