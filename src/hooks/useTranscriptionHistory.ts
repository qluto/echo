import { useState, useCallback } from "react";
import {
  getTranscriptionHistory,
  searchTranscriptionHistory,
  deleteTranscriptionEntry,
  clearTranscriptionHistory,
  TranscriptionHistoryEntry,
  TranscriptionHistoryPage,
} from "../lib/tauri";

interface UseTranscriptionHistoryReturn {
  entries: TranscriptionHistoryEntry[];
  totalCount: number;
  hasMore: boolean;
  isLoading: boolean;
  searchQuery: string;
  error: string | null;
  loadHistory: (reset?: boolean) => Promise<void>;
  loadMore: () => Promise<void>;
  search: (query: string) => Promise<void>;
  clearSearch: () => Promise<void>;
  remove: (id: number) => Promise<void>;
  clearAll: () => Promise<void>;
  refresh: () => Promise<void>;
}

const PAGE_SIZE = 20;

export function useTranscriptionHistory(): UseTranscriptionHistoryReturn {
  const [entries, setEntries] = useState<TranscriptionHistoryEntry[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [error, setError] = useState<string | null>(null);

  const fetchPage = useCallback(
    async (query: string, offset: number): Promise<TranscriptionHistoryPage> => {
      if (query) {
        return searchTranscriptionHistory(query, PAGE_SIZE, offset);
      }
      return getTranscriptionHistory(PAGE_SIZE, offset);
    },
    []
  );

  const loadHistory = useCallback(
    async (reset = true) => {
      try {
        setIsLoading(true);
        setError(null);
        const page = await fetchPage(searchQuery, 0);
        if (reset) {
          setEntries(page.entries);
        } else {
          setEntries((prev) => [...prev, ...page.entries]);
        }
        setTotalCount(page.total_count);
        setHasMore(page.has_more);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setIsLoading(false);
      }
    },
    [searchQuery, fetchPage]
  );

  const loadMore = useCallback(async () => {
    if (!hasMore || isLoading) return;
    try {
      setIsLoading(true);
      const page = await fetchPage(searchQuery, entries.length);
      setEntries((prev) => [...prev, ...page.entries]);
      setHasMore(page.has_more);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsLoading(false);
    }
  }, [hasMore, isLoading, searchQuery, entries.length, fetchPage]);

  const search = useCallback(
    async (query: string) => {
      setSearchQuery(query);
      try {
        setIsLoading(true);
        setError(null);
        const page = query
          ? await searchTranscriptionHistory(query, PAGE_SIZE, 0)
          : await getTranscriptionHistory(PAGE_SIZE, 0);
        setEntries(page.entries);
        setTotalCount(page.total_count);
        setHasMore(page.has_more);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const clearSearch = useCallback(async () => {
    await search("");
  }, [search]);

  const remove = useCallback(async (id: number) => {
    try {
      await deleteTranscriptionEntry(id);
      setEntries((prev) => prev.filter((e) => e.id !== id));
      setTotalCount((c) => Math.max(0, c - 1));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const clearAll = useCallback(async () => {
    try {
      await clearTranscriptionHistory();
      setEntries([]);
      setTotalCount(0);
      setHasMore(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const refresh = useCallback(async () => {
    await loadHistory(true);
  }, [loadHistory]);

  return {
    entries,
    totalCount,
    hasMore,
    isLoading,
    searchQuery,
    error,
    loadHistory,
    loadMore,
    search,
    clearSearch,
    remove,
    clearAll,
    refresh,
  };
}
