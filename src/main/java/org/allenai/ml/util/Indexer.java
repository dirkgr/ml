package org.allenai.ml.util;

import com.gs.collections.api.map.primitive.ObjectDoubleMap;
import com.gs.collections.api.map.primitive.ObjectIntMap;
import com.gs.collections.impl.map.mutable.primitive.ObjectDoubleHashMap;
import com.gs.collections.impl.map.mutable.primitive.ObjectIntHashMap;
import lombok.val;
import org.allenai.ml.linalg.Vector;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * A list that implements O(1) random access lookup AND O(1) <code>indexOf</code> calls. It also
 * does not permit duplicate elements.
 * @param <T>
 */
public class Indexer<T extends Comparable<T>> extends AbstractList<T> {

    private final List<T> list;
    private final ObjectIntMap<T> objToIndex;

    @Override
    public boolean equals(Object o) {
        val r = innerEquals(o);
        if(!r)
            System.err.println(this.getClass().getName() + " not equals");
        return r;
    }

    private boolean innerEquals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;

        Indexer<?> indexer = (Indexer<?>) o;

        if (!list.equals(indexer.list)) return false;
        return objToIndex.equals(indexer.objToIndex);

    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + list.hashCode();
        result = 31 * result + objToIndex.hashCode();
        return result;
    }

    private Indexer(Stream<T> elems) {
        this.list = elems
            .distinct()
            .collect(Collectors.toList());
        val m = new ObjectIntHashMap<T>();
        for (int idx = 0; idx < list.size(); idx++) {
            m.put(list.get(idx), idx);
        }
        this.objToIndex = m.toImmutable();
    }

    @Override
    public int size() {
        return list.size();
    }

    @Override
    public boolean isEmpty() {
        return list.isEmpty();
    }

    @Override
    public boolean contains(Object o) {
        return objToIndex.containsKey(o);
    }

    @Override
    public Iterator<T> iterator() {
        return list.iterator();
    }

    @Override
    public Object[] toArray() {
        return list.toArray();
    }

    @Override
    public boolean add(T t) {
        throw new RuntimeException("Indexer is immutable and doesn't support add()");
    }

    @Override
    public T get(int index) {
        return list.get(index);
    }

    @Override
    public int indexOf(Object o) {
        return objToIndex.get(o);
    }

    @Override
    public int lastIndexOf(Object o) {
        return indexOf(o);
    }


    @Override
    public List<T> subList(int fromIndex, int toIndex) {
        return list.subList(fromIndex, toIndex);
    }


    public static <T extends Comparable<T>> Indexer<T> fromStream(Stream<T> stream) {
        return new Indexer<>(stream);
    }

    public ObjectDoubleMap<T> toMap(Vector vector) {
        ObjectDoubleHashMap<T> m = new ObjectDoubleHashMap<T>();
        vector.nonZeroEntries().forEach(e -> {
            m.put(this.get((int)e.index), e.value);
        });
        return m;
    }

    private final static String DATA_VERSION = "2.0";

    public void save(DataOutputStream dos) throws IOException {
        dos.writeUTF(DATA_VERSION);
        if (list.size() != objToIndex.size()) {
            throw new RuntimeException("Trying to save indexer with list/set size mismatch");
        }
        IOUtils.saveList(dos, list.stream().map(Object::toString).collect(Collectors.toList()));
    }

    public static Indexer<String> load(DataInputStream dis) throws IOException {
        IOUtils.ensureVersionMatch(dis, DATA_VERSION);
        val lst = IOUtils.loadList(dis);
        return new Indexer<>(lst.stream());
    }

    public static Indexer<String> loadOldVersion(DataInputStream dis) throws IOException {
        IOUtils.ensureVersionMatch(dis, "1.0");
        List lst = IOUtils.loadListOldVersion(dis);
        return new Indexer(lst.stream());
    }
}
