import { expect, test } from 'bun:test'
import { pickOnEnter } from './pickOnEnter'

const opts = [{ value: 'cell_type_v2' }, { value: 'cell_type' }]

test('exact match wins', () => {
  expect(pickOnEnter(opts, 'cell_type', true)?.value).toBe('cell_type')
})

test('a new name is created, not swallowed by a lone fuzzy match', () => {
  expect(pickOnEnter([opts[0]], 'cell_type', true)).toBeNull()
})

test('a lone fuzzy match is taken when nothing can be created', () => {
  expect(pickOnEnter([opts[0]], 'cell', false)?.value).toBe('cell_type_v2')
})

test('several fuzzy matches pick nothing', () => {
  expect(pickOnEnter(opts, 'cell', false)).toBeNull()
})
