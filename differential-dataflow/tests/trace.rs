use timely::container::PushInto;
use timely::dataflow::operators::generic::OperatorInfo;
use timely::progress::{Antichain, frontier::AntichainRef};

use differential_dataflow::trace::implementations::{ValBatcher, ValBuilder, ValSpine};
use differential_dataflow::trace::{Trace, TraceReader, Batcher, Builder};
use differential_dataflow::trace::cursor::Cursor;

type IntegerTrace = ValSpine<u64, u64, usize, i64>;
type IntegerBuilder = ValBuilder<u64, u64, usize, i64>;

fn get_trace() -> ValSpine<u64, u64, usize, i64> {
    let op_info = OperatorInfo::new(0, 0, [].into());
    let mut trace = IntegerTrace::new(op_info, None, None);
    {
        let mut batcher = ValBatcher::<u64,u64,usize,i64>::new(None, 0);

        batcher.push_into(vec![
            ((1, 2), 0, 1),
            ((2, 3), 1, 1),
            ((2, 3), 2, -1),
        ]);

        let batch_ts = &[1, 2, 3];
        for i in batch_ts {
            let (mut chain, description) = batcher.seal(Antichain::from_elem(*i));
            trace.insert(IntegerBuilder::seal(&mut chain, description));
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

/// Regression: `CursorList::seek_val` must only dispatch to sub-cursors positioned on a valid key.
///
/// After `seek_key(2)` the sub-cursor for the second batch (which lacks key 2) is past its end;
/// dispatching `seek_val` to it indexed out of range.
#[test]
fn seek_val_with_multiple_batches_does_not_panic() {
    let op_info = OperatorInfo::new(0, 0, [].into());
    let mut trace = IntegerTrace::new(op_info, None, None);

    let mut batcher = ValBatcher::<u64,u64,usize,i64>::new(None, 0);
    batcher.push_into(vec![
        ((1, 10), 0, 1),
        ((2, 20), 0, 1),
        ((1, 30), 1, 1),
    ]);
    for upper in [1, 2] {
        let (mut chain, description) = batcher.seal(Antichain::from_elem(upper));
        trace.insert(IntegerBuilder::seal(&mut chain, description));
    }

    let (mut cursor, storage) = trace.cursor();
    cursor.seek_key(&storage, &2);
    assert!(cursor.key_valid(&storage));
    assert_eq!(*cursor.key(&storage), 2);

    cursor.seek_val(&storage, &20);
    assert!(cursor.val_valid(&storage));
    assert_eq!(*cursor.val(&storage), 20);
}
