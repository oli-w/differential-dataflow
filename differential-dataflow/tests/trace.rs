use timely::dataflow::operators::generic::OperatorInfo;
use timely::progress::{Antichain, frontier::AntichainRef};

use differential_dataflow::trace::implementations::{ValBatcher, ValBuilder, ValSpine};
use differential_dataflow::trace::{Trace, TraceReader, Batcher};
use differential_dataflow::trace::cursor::Cursor;

type IntegerTrace = ValSpine<u64, u64, usize, i64>;
type IntegerBuilder = ValBuilder<u64, u64, usize, i64>;

fn get_trace() -> ValSpine<u64, u64, usize, i64> {
    let op_info = OperatorInfo::new(0, 0, [].into());
    let mut trace = IntegerTrace::new(op_info, None, None);
    {
        let mut batcher = ValBatcher::<u64,u64,usize,i64>::new(None, 0);

        batcher.push_container(&mut vec![
            ((1, 2), 0, 1),
            ((2, 3), 1, 1),
            ((2, 3), 2, -1),
        ]);

        let batch_ts = &[1, 2, 3];
        let batches = batch_ts.iter().map(move |i| batcher.seal::<IntegerBuilder>(Antichain::from_elem(*i)));
        for b in batches {
            trace.insert(b);
        }
    }
    trace
}

#[test]
fn test_trace() {
    let mut trace = get_trace();

    let (mut cursor1, storage1) = trace.cursor_through(AntichainRef::new(&[1])).unwrap();
    let vec_1 = cursor1.to_vec(&storage1, |k| k.clone(), |v| v.clone());
    assert_eq!(vec_1, vec![((1, 2), vec![(0, 1)])]);

    let (mut cursor2, storage2) = trace.cursor_through(AntichainRef::new(&[2])).unwrap();
    let vec_2 = cursor2.to_vec(&storage2, |k| k.clone(), |v| v.clone());
    println!("--> {:?}", vec_2);
    assert_eq!(vec_2, vec![
               ((1, 2), vec![(0, 1)]),
               ((2, 3), vec![(1, 1)]),
    ]);

    let (mut cursor3, storage3) = trace.cursor_through(AntichainRef::new(&[3])).unwrap();
    let vec_3 = cursor3.to_vec(&storage3, |k| k.clone(), |v| v.clone());
    assert_eq!(vec_3, vec![
               ((1, 2), vec![(0, 1)]),
               ((2, 3), vec![(1, 1), (2, -1)]),
    ]);

    let (mut cursor4, storage4) = trace.cursor();
    let vec_4 = cursor4.to_vec(&storage4, |k| k.clone(), |v| v.clone());
    assert_eq!(vec_4, vec_3);
}

/// Regression: CursorList::seek_val used to dispatch to ALL sub-cursors,
/// including those where key_valid is false (positioned past the end by a
/// prior seek_key). OrdValCursor::seek_val then called bounds() on an
/// out-of-range key_cursor, panicking with an OffsetList index-out-of-bounds.
///
/// The fix: seek_val should only dispatch to min_key cursors (those with a
/// valid key), consistent with step_val and rewind_vals.
#[test]
fn seek_val_with_multiple_batches_does_not_panic() {
    let op_info = OperatorInfo::new(0, 0, [].into());
    let mut trace = IntegerTrace::new(op_info, None, None);

    // Push all data into one batcher, then seal two separate batches at
    // increasing frontiers. This gives us:
    //   Batch 0 [time 0→1): key 1 val 10, key 2 val 20
    //   Batch 1 [time 1→2): key 1 val 30           (key 2 absent)
    let mut batcher = ValBatcher::<u64, u64, usize, i64>::new(None, 0);
    batcher.push_container(&mut vec![
        ((1, 10), 0, 1),
        ((2, 20), 0, 1),
        ((1, 30), 1, 1),
    ]);

    let batch_0 = batcher.seal::<IntegerBuilder>(Antichain::from_elem(1));
    trace.insert(batch_0);
    let batch_1 = batcher.seal::<IntegerBuilder>(Antichain::from_elem(2));
    trace.insert(batch_1);

    // cursor() over the full trace gives a CursorList with 2 sub-cursors.
    // seek_key(2) positions sub-cursor 0 on key 2, but sub-cursor 1 past
    // the end (key_valid=false). seek_val must not dispatch to sub-cursor 1.
    let (mut cursor, storage) = trace.cursor();
    cursor.seek_key(&storage, &2);
    assert!(cursor.key_valid(&storage));
    assert_eq!(*cursor.key(&storage), 2);

    // This panicked before the fix: seek_val dispatched to the sub-cursor
    // with key_valid=false, which called bounds() on an out-of-range index.
    cursor.seek_val(&storage, &20);
    assert!(cursor.val_valid(&storage));
    assert_eq!(*cursor.val(&storage), 20);
}
