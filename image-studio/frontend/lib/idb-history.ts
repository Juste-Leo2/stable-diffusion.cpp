// IndexedDB persistence for `useHistory`. Stores the raw image Blob (not a
// throwaway object URL) so history survives page reload.

const DB_NAME = "bonsai-image-studio";
const STORE = "history";
const VERSION = 1;

let _dbPromise: Promise<IDBDatabase> | null = null;

function ensureDb(): Promise<IDBDatabase> | null {
  if (typeof indexedDB === "undefined") return null;
  if (_dbPromise) return _dbPromise;
  _dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "id" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return _dbPromise;
}

export interface StoredEntry {
  id: string;
  prompt: string;
  params: { seed: number; steps: number; backend: string; resolutionId: string };
  imageBlob: Blob;
  timestamp: number;
}

export async function idbPut(entry: StoredEntry): Promise<void> {
  const dbP = ensureDb();
  if (!dbP) return;
  const db = await dbP;
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).put(entry);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error ?? new Error("transaction aborted"));
  });
}

export async function idbDelete(id: string): Promise<void> {
  const dbP = ensureDb();
  if (!dbP) return;
  const db = await dbP;
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error ?? new Error("transaction aborted"));
  });
}

export async function idbClear(): Promise<void> {
  const dbP = ensureDb();
  if (!dbP) return;
  const db = await dbP;
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error ?? new Error("transaction aborted"));
  });
}

export async function idbGetAll(): Promise<StoredEntry[]> {
  const dbP = ensureDb();
  if (!dbP) return [];
  const db = await dbP;
  return new Promise<StoredEntry[]>((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const req = tx.objectStore(STORE).getAll();
    req.onsuccess = () => resolve((req.result as StoredEntry[]) ?? []);
    req.onerror = () => reject(req.error);
    tx.onabort = () => reject(tx.error ?? new Error("transaction aborted"));
  });
}
