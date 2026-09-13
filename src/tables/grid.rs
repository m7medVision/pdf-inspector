//! Column/row boundary detection and cell assignment for heuristic tables.

use crate::types::TextItem;

use super::{Table, TableDetectionMode};

pub(crate) fn find_column_boundaries(
    items: &[(usize, &TextItem)],
    mode: TableDetectionMode,
) -> Vec<f32> {
    let mut x_positions: Vec<f32> = items.iter().map(|(_, i)| i.x).collect();
    x_positions.sort_by(|a, b| a.total_cmp(b));

    if x_positions.is_empty() {
        return vec![];
    }

    // For dense, narrow-column tables (e.g. train schedules with 24 cols at
    // 26pt spacing), the old avg_gap approach over-clusters because avg_gap is
    // dominated by the many items *within* each column.  Use a gap-histogram
    // on consecutive position gaps to detect when columns are densely packed,
    // and only then lower the threshold below 25pt.
    let x_range = x_positions.last().unwrap() - x_positions.first().unwrap();
    let avg_gap = if x_positions.len() > 1 {
        x_range / (x_positions.len() - 1) as f32
    } else {
        60.0
    };

    // Default: original avg_gap approach, center-based clustering
    let mut cluster_threshold = avg_gap.clamp(25.0, 50.0);
    let mut use_edge_clustering = false;

    // Analyze the distribution of non-trivial consecutive gaps to detect
    // a bimodal pattern (small within-column gaps vs large between-column gaps).
    // When detected, switch to edge-based clustering with the lower threshold
    // to correctly separate densely-packed columns without over-splitting
    // wide columns (edge-based avoids the center-drift problem).
    let mut consec_gaps: Vec<f32> = x_positions
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&g| g > 0.1) // skip near-duplicate positions
        .collect();

    if consec_gaps.len() > 2 {
        consec_gaps.sort_by(|a, b| a.total_cmp(b));
        // Find the biggest jump in the sorted gap sequence — natural break
        // between within-column jitter and between-column spacing.
        // Require at least 3 values on each side to avoid outlier-dominated
        // splits (e.g. a single large page-margin gap).
        let mut best_split = consec_gaps.len() / 2;
        let mut best_jump = 0.0f32;
        let min_side = 3.min(consec_gaps.len() / 2);
        for i in 0..consec_gaps.len().saturating_sub(1) {
            let left_count = i + 1;
            let right_count = consec_gaps.len() - i - 1;
            if left_count < min_side || right_count < min_side {
                continue;
            }
            let jump = consec_gaps[i + 1] - consec_gaps[i];
            if jump > best_jump {
                best_jump = jump;
                best_split = i;
            }
        }
        let threshold = (consec_gaps[best_split]
            + consec_gaps[(best_split + 1).min(consec_gaps.len() - 1)])
            / 2.0;
        // Override for tables with a clear bimodal gap pattern:
        // - Dense tables (500+ items, e.g. 24-column train schedule): use
        //   edge-based clustering with the detected threshold.
        // - Smaller tables with a strong bimodal signal (jump > 10pt):
        //   lower the threshold but keep center-based clustering to avoid
        //   over-splitting.
        if threshold < 15.0 && best_jump > 2.0 && x_positions.len() > 500 {
            cluster_threshold = threshold.clamp(8.0, 25.0);
            use_edge_clustering = true;
        } else if best_jump > 10.0 && threshold < cluster_threshold {
            // Strong bimodal signal even with fewer items — the gap between
            // within-column jitter and between-column spacing is unambiguous.
            cluster_threshold = threshold.max(8.0);
        }
    }

    // Track cluster membership: for each cluster, store the list of x positions
    let mut cluster_xs: Vec<Vec<f32>> = vec![vec![x_positions[0]]];

    for &x in &x_positions[1..] {
        let last_cluster = cluster_xs.last().unwrap();
        // For dense columns (gap-histogram triggered), use edge-based clustering:
        // compare with the last item to avoid center-drift that merges adjacent
        // narrow columns.  For normal tables, use center-based (original behavior).
        let reference = if use_edge_clustering {
            *last_cluster.last().unwrap()
        } else {
            last_cluster.iter().sum::<f32>() / last_cluster.len() as f32
        };

        if x - reference > cluster_threshold {
            cluster_xs.push(vec![x]);
        } else {
            cluster_xs.last_mut().unwrap().push(x);
        }
    }

    // Numeric column merge pass: when a sparse cluster (few items, typically
    // header text) is adjacent to a dense numeric cluster and within 1.5×
    // threshold, merge them. This fixes tables where multi-line wrapped
    // headers have slightly different X positions than the data columns,
    // causing the header and data to split into separate clusters.
    let columns_before_merge = cluster_xs.len();
    if columns_before_merge >= 3 {
        cluster_xs = merge_numeric_adjacent_clusters(cluster_xs, items, cluster_threshold);
    }

    let columns: Vec<f32> = cluster_xs
        .iter()
        .map(|xs| xs.iter().sum::<f32>() / xs.len() as f32)
        .collect();

    // Filter columns - each should have multiple items
    let min_items_per_col = (items.len() / columns.len().max(1) / 4).max(2);
    let columns: Vec<f32> = columns
        .into_iter()
        .filter(|&col_x| {
            items
                .iter()
                .filter(|(_, i)| (i.x - col_x).abs() < cluster_threshold)
                .count()
                >= min_items_per_col
        })
        .collect();

    log::debug!(
        "  find_column_boundaries: {} columns (merged from {}), threshold={:.1}, {} items",
        columns.len(),
        columns_before_merge,
        cluster_threshold,
        items.len()
    );

    // Anti-paragraph safeguard for BodyFont mode:
    // Paragraphs concentrate items at the left margin; tables distribute evenly.
    // Reject if any single column has >60% of all items.
    if mode == TableDetectionMode::BodyFont {
        let total_items = items.len();
        for &col_x in &columns {
            let count = items
                .iter()
                .filter(|(_, i)| (i.x - col_x).abs() < cluster_threshold)
                .count();
            if count as f32 / total_items as f32 > 0.60 {
                return vec![];
            }
        }
    }

    columns
}

/// Check if a text string looks like a number (digits, decimals, sign, comma).
fn is_numeric_text(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    // Match patterns like: 8.23, -1.05, 9.99, 7.12, 100, 3,456.78, +5%, ---
    // But NOT: BIO, Department, Core Courses
    s.chars()
        .all(|c| c.is_ascii_digit() || c == '.' || c == ',' || c == '-' || c == '+' || c == '%')
        && s.chars().any(|c| c.is_ascii_digit())
}

/// Merge adjacent X-position clusters when one is a sparse header cluster
/// and the other is a dense numeric data cluster. This prevents multi-line
/// wrapped headers from splitting a logical column into two clusters.
fn merge_numeric_adjacent_clusters(
    mut clusters: Vec<Vec<f32>>,
    items: &[(usize, &TextItem)],
    threshold: f32,
) -> Vec<Vec<f32>> {
    // For each cluster, compute: center, item count, numeric fraction
    struct ClusterInfo {
        center: f32,
        count: usize,
        numeric_frac: f32,
    }

    let compute_info = |xs: &[f32]| -> ClusterInfo {
        let center = xs.iter().sum::<f32>() / xs.len() as f32;
        // Count items and numeric fraction for items near this cluster center
        let mut total = 0;
        let mut numeric = 0;
        for (_, item) in items {
            if (item.x - center).abs() < threshold {
                total += 1;
                if is_numeric_text(&item.text) {
                    numeric += 1;
                }
            }
        }
        ClusterInfo {
            center,
            count: total,
            numeric_frac: if total > 0 {
                numeric as f32 / total as f32
            } else {
                0.0
            },
        }
    };

    // Merge distance: allow merging clusters that are slightly beyond the
    // original threshold. Use 1.5× threshold to catch header-vs-data splits.
    let merge_dist = threshold * 1.5;

    // Iterate and merge adjacent pairs. Use a simple left-to-right scan.
    let mut merged = true;
    while merged {
        merged = false;
        let mut i = 0;
        while i + 1 < clusters.len() {
            let info_a = compute_info(&clusters[i]);
            let info_b = compute_info(&clusters[i + 1]);
            let dist = (info_b.center - info_a.center).abs();

            if dist > merge_dist {
                i += 1;
                continue;
            }

            // Determine if one cluster is sparse (header) and the other
            // is dense and numeric (data). A cluster is "sparse" if it has
            // significantly fewer items than the other.
            let (sparse, dense) = if info_a.count < info_b.count {
                (&info_a, &info_b)
            } else {
                (&info_b, &info_a)
            };

            // Merge if the dense cluster is predominantly numeric (>50%)
            // and the sparse cluster has at most 1/3 the items of the dense one.
            let should_merge =
                dense.numeric_frac > 0.50 && sparse.count <= dense.count / 2 && sparse.count <= 5;

            if should_merge {
                log::debug!(
                    "  merging column clusters: center {:.1} ({} items, {:.0}% numeric) + {:.1} ({} items, {:.0}% numeric), dist={:.1}",
                    info_a.center,
                    info_a.count,
                    info_a.numeric_frac * 100.0,
                    info_b.center,
                    info_b.count,
                    info_b.numeric_frac * 100.0,
                    dist,
                );
                // Merge cluster i+1 into cluster i
                let next = clusters.remove(i + 1);
                clusters[i].extend(next);
                merged = true;
                // Don't increment i — check if the merged cluster can merge further
            } else {
                i += 1;
            }
        }
    }

    clusters
}

/// Find row boundaries by clustering Y positions
pub(crate) fn find_row_boundaries(items: &[(usize, &TextItem)]) -> Vec<f32> {
    let mut y_positions: Vec<f32> = items.iter().map(|(_, i)| i.y).collect();
    y_positions.sort_by(|a, b| b.total_cmp(a)); // Descending

    if y_positions.is_empty() {
        return vec![];
    }

    // Cluster Y positions - items within a fraction of the median font size are same row.
    // Using 0.8× median font keeps the threshold between intra-row gaps (~0pt) and
    // inter-row gaps (≥1× font size), preventing row merging in uniform-spaced PDFs.
    let cluster_threshold = {
        let mut font_sizes: Vec<f32> = items.iter().map(|(_, i)| i.font_size).collect();
        font_sizes.sort_by(|a, b| a.total_cmp(b));
        let median_font = font_sizes[font_sizes.len() / 2];
        (median_font * 0.8).max(4.0)
    };
    let mut rows = Vec::new();
    let mut cluster_items: Vec<f32> = vec![y_positions[0]];

    for &y in &y_positions[1..] {
        let cluster_center = cluster_items.iter().sum::<f32>() / cluster_items.len() as f32;

        if cluster_center - y >= cluster_threshold {
            // End current cluster (note: Y is descending)
            rows.push(cluster_center);
            cluster_items = vec![y];
        } else {
            cluster_items.push(y);
        }
    }

    if !cluster_items.is_empty() {
        rows.push(cluster_items.iter().sum::<f32>() / cluster_items.len() as f32);
    }

    rows
}

/// Find which column index an X position belongs to
pub(crate) fn find_column_index(columns: &[f32], x: f32) -> Option<usize> {
    // Calculate adaptive threshold based on column spacing
    let threshold = if columns.len() >= 2 {
        let min_gap = columns
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(f32::INFINITY, f32::min);
        (min_gap / 2.0).clamp(25.0, 50.0)
    } else {
        50.0
    };

    columns
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (x - *a)
                .abs()
                .partial_cmp(&(x - *b).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .filter(|(_, col_x)| (x - *col_x).abs() < threshold)
        .map(|(idx, _)| idx)
}

/// Find which row index a Y position belongs to
pub(crate) fn find_row_index(rows: &[f32], y: f32) -> Option<usize> {
    let threshold = 15.0;
    rows.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (y - *a)
                .abs()
                .partial_cmp(&(y - *b).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .filter(|(_, row_y)| (y - *row_y).abs() < threshold)
        .map(|(idx, _)| idx)
}

/// Recover a header row for small-font tables by looking at body-font items
/// just above the table's first row.
///
/// PDF tables often have header rows at the body font size while data rows use
/// a smaller font. Pass 1 (SmallFont) excludes the header because of the
/// font-size filter. This function looks upward from the table's first row for
/// body-font items that align with the table's columns, and prepends them.
pub(crate) fn recover_header_row(
    table: &mut Table,
    all_items: &[TextItem],
    small_font_threshold: f32,
) {
    if table.rows.is_empty() || table.columns.is_empty() {
        return;
    }

    let first_row_y = table.rows[0]; // highest Y (rows are descending)

    // Compute typical row spacing for gap threshold
    let row_gap_limit = if table.rows.len() >= 2 {
        let avg_spacing =
            (table.rows[0] - table.rows[table.rows.len() - 1]) / (table.rows.len() - 1) as f32;
        // Allow up to 2x average row spacing for the header gap
        (avg_spacing * 2.0).clamp(10.0, 40.0)
    } else {
        30.0
    };

    // Find body-font items just above the first row
    let header_candidates: Vec<(usize, &TextItem)> = all_items
        .iter()
        .enumerate()
        .filter(|(_, item)| {
            !item.is_strikeout
                && item.font_size > small_font_threshold
                && item.y > first_row_y
                && item.y <= first_row_y + row_gap_limit
        })
        .collect();

    if header_candidates.is_empty() {
        return;
    }

    // Group header candidates by Y (cluster within 5pt)
    let mut header_y_groups: Vec<(f32, Vec<(usize, &TextItem)>)> = Vec::new();
    let mut sorted_candidates = header_candidates;
    sorted_candidates.sort_by(|a, b| {
        b.1.y
            .partial_cmp(&a.1.y)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (idx, item) in &sorted_candidates {
        let found = header_y_groups
            .iter_mut()
            .find(|(y, _)| (item.y - *y).abs() < 5.0);
        if let Some((_, group)) = found {
            group.push((*idx, item));
        } else {
            header_y_groups.push((item.y, vec![(*idx, item)]));
        }
    }

    // Take the row closest to the table (lowest Y above first_row_y)
    // header_y_groups is sorted by descending Y, so take the last one
    let (header_y, header_items) = header_y_groups.last().unwrap();

    // Map header items to table columns
    let num_cols = table.columns.len();
    let mut header_cells: Vec<String> = vec![String::new(); num_cols];
    let mut mapped_count = 0;
    let mut header_indices = Vec::new();

    for (idx, item) in header_items {
        if let Some(col) = find_column_index(&table.columns, item.x) {
            let text = item.text.trim();
            if !text.is_empty() {
                if !header_cells[col].is_empty() {
                    header_cells[col].push(' ');
                }
                header_cells[col].push_str(text);
                mapped_count += 1;
                header_indices.push(*idx);
            }
        }
    }

    // Require at least 2 columns populated to look like a real header row
    let populated = header_cells.iter().filter(|c| !c.is_empty()).count();
    if populated < 2 || mapped_count < 2 {
        return;
    }

    // Prepend header row to the table
    table.rows.insert(0, *header_y);
    table.cells.insert(0, header_cells);
    table.item_indices.extend(header_indices);
}

const MAX_HEADER_LINES: usize = 8;

/// Recover column headers that wrap onto several lines above a table's first
/// row and were left out of it.
///
/// A header set in small type over narrow columns ("Retained earnings" over
/// "(accumulated losses)" over a period split after its dash) stacks lines
/// closer together than the table's rows, with the lines of neighbouring
/// columns on staggered baselines. Row clustering cannot make one row of
/// that, so the detector drops those rows and they fall back into the text
/// flow as a paragraph. Here each line above the table is assigned to a
/// column by where it sits over the body text, and the lines are joined top
/// to bottom within their column into one header row.
///
/// The block is the run of lines directly above the first row, no further
/// apart than the rows themselves, in which every item lands in exactly one
/// column. Label-only lines between the header and the first row are section
/// rows and join the body. Nothing changes unless the table's values are
/// figures and the header reaches at least two data columns and half of
/// them. Columns left with no text in any row, cluster positions seeded by
/// the header's left edges, are dropped.
pub(crate) fn recover_wrapped_column_headers(
    table: &mut Table,
    items: &[TextItem],
    claimed: &std::collections::HashSet<usize>,
) {
    let column_count = table.columns.len();
    if column_count < 3 || table.rows.len() < 2 {
        return;
    }

    let mut extents: Vec<Option<(f32, f32)>> = vec![None; column_count];
    let mut font_sizes = Vec::new();
    for &index in &table.item_indices {
        let item = &items[index];
        if let Some(column) = find_column_index(&table.columns, item.x) {
            let extent = extents[column].get_or_insert((item.x, item.x + item.width));
            extent.0 = extent.0.min(item.x);
            extent.1 = extent.1.max(item.x + item.width);
        }
        font_sizes.push(item.font_size);
    }
    let data_columns = extents.iter().skip(1).filter(|e| e.is_some()).count();
    if extents[0].is_none() || data_columns < 2 || font_sizes.is_empty() {
        return;
    }
    // Only figures under labels: in a table of wrapped prose the lines above
    // the first row are as likely the previous rows of another grid.
    let values: Vec<&str> = table
        .cells
        .iter()
        .flat_map(|row| row.iter().skip(1))
        .map(|cell| cell.trim())
        .filter(|cell| !cell.is_empty())
        .collect();
    let figures = values
        .iter()
        .filter(|v| is_numeric_text(v.trim_start_matches('(').trim_end_matches(')')))
        .count();
    if values.is_empty() || (figures as f32) < values.len() as f32 * 0.8 {
        return;
    }
    font_sizes.sort_by(|a, b| a.total_cmp(b));
    let font_size = font_sizes[font_sizes.len() / 2];
    if font_size <= 0.0 {
        return;
    }
    let mut pitches: Vec<f32> = table.rows.windows(2).map(|w| w[0] - w[1]).collect();
    pitches.sort_by(|a, b| a.total_cmp(b));
    let max_gap = (pitches[pitches.len() / 2] * 1.25).max(font_size * 1.5);
    // Columns split at the middle of the gap between their body texts. A
    // header is centred on, left-aligned over or right-aligned over its
    // column's values, so its centre lands in that column's span, but it
    // must not reach over another column's values: that is a spanning
    // header, which has no single column.
    let present: Vec<(usize, f32, f32)> = extents
        .iter()
        .enumerate()
        .filter_map(|(c, e)| e.map(|(a, b)| (c, a, b)))
        .collect();
    let column_of = |item: &TextItem| -> Option<usize> {
        let (left, right) = (item.x, item.x + item.width);
        let centre = (left + right) / 2.0;
        let slot = present
            .windows(2)
            .position(|w| centre < (w[0].2 + w[1].1) / 2.0);
        let (column, _, _) = present[slot.unwrap_or(present.len() - 1)];
        let reaches_other = present
            .iter()
            .any(|&(c, a, b)| c != column && right > a && left < b);
        (!reaches_other).then_some(column)
    };

    let in_table: std::collections::HashSet<usize> = table.item_indices.iter().copied().collect();
    let first_row_y = table.rows[0];
    let mut candidates: Vec<usize> = (0..items.len())
        .filter(|i| {
            let item = &items[*i];
            !in_table.contains(i)
                && !claimed.contains(i)
                && !item.text.trim().is_empty()
                && item.is_upright()
                && item.line_y() > first_row_y + font_size * 0.3
                && item.font_size >= font_size * 0.75
                && item.font_size <= font_size * 1.35
        })
        .collect();
    candidates.sort_by(|a, b| items[*a].line_y().total_cmp(&items[*b].line_y()));

    // Lines ascending from the table, each a list of (item, column).
    let mut lines: Vec<(f32, Vec<(usize, usize)>)> = Vec::new();
    let mut previous_y = first_row_y;
    let mut rest = candidates.as_slice();
    while let Some(&first) = rest.first() {
        let y = items[first].line_y();
        if y - previous_y > max_gap {
            break;
        }
        let end = rest
            .iter()
            .position(|&i| items[i].line_y() - y > font_size * 0.3)
            .unwrap_or(rest.len());
        let Some(line) = rest[..end]
            .iter()
            .map(|&i| column_of(&items[i]).map(|c| (i, c)))
            .collect::<Option<Vec<_>>>()
        else {
            break;
        };
        lines.push((y, line));
        previous_y = y;
        rest = &rest[end..];
        if lines.len() == MAX_HEADER_LINES {
            break;
        }
    }

    let label_only = |line: &[(usize, usize)]| line.iter().all(|&(_, c)| c == 0);
    let section_count = lines.iter().take_while(|(_, l)| label_only(l)).count();
    while lines.len() > section_count && lines.last().is_some_and(|(_, l)| label_only(l)) {
        lines.pop();
    }
    let header_lines = &lines[section_count..];
    let mut header_parts: Vec<Vec<&TextItem>> = vec![Vec::new(); column_count];
    for &(index, column) in header_lines.iter().flat_map(|(_, l)| l) {
        header_parts[column].push(&items[index]);
    }
    let headed = header_parts
        .iter()
        .skip(1)
        .filter(|p| !p.is_empty())
        .count();
    if headed < 2 || headed * 2 < data_columns {
        return;
    }

    let mut header = Vec::with_capacity(column_count);
    for parts in &mut header_parts {
        parts.sort_by(|a, b| b.line_y().total_cmp(&a.line_y()).then(a.x.total_cmp(&b.x)));
        let mut text = String::new();
        for part in parts.iter() {
            if !text.is_empty() && !text.ends_with('-') {
                text.push(' ');
            }
            text.push_str(part.text.trim());
        }
        header.push(text);
    }

    for (y, line) in lines[..section_count].iter() {
        let mut parts: Vec<&TextItem> = line.iter().map(|&(i, _)| &items[i]).collect();
        parts.sort_by(|a, b| a.x.total_cmp(&b.x));
        let mut row = vec![String::new(); column_count];
        row[0] = parts
            .iter()
            .map(|p| p.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        table.rows.insert(0, *y);
        table.cells.insert(0, row);
    }
    let header_y = header_lines.last().map_or(first_row_y, |(y, _)| *y);
    table.rows.insert(0, header_y);
    table.cells.insert(0, header);
    table
        .item_indices
        .extend(lines.iter().flat_map(|(_, l)| l.iter().map(|&(i, _)| i)));

    let keep: Vec<bool> = (0..column_count)
        .map(|c| table.cells.iter().any(|row| !row[c].trim().is_empty()))
        .collect();
    if keep.iter().any(|k| !k) {
        let mut column = 0;
        table.columns.retain(|_| {
            column += 1;
            keep[column - 1]
        });
        for row in &mut table.cells {
            let mut column = 0;
            row.retain(|_| {
                column += 1;
                keep[column - 1]
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::TableKind;
    use crate::types::ItemType;

    fn make_item(text: &str, x: f32, y: f32, font_size: f32) -> TextItem {
        TextItem {
            text: text.to_string(),
            x,
            y,
            width: text.len() as f32 * font_size * 0.5,
            height: font_size,
            font: "TestFont".to_string(),
            font_tag: String::new(),
            legacy_symbol_rewrite: false,
            font_size,
            page: 1,
            is_bold: false,
            is_italic: false,
            is_underline: false,
            is_strikeout: false,
            rotation: 0.0,
            advance_known: true,
            item_type: ItemType::Text,
            mcid: None,
            baseline_shift: 0.0,
        }
    }

    // --- find_column_index ---

    #[test]
    fn test_find_column_index_exact_match() {
        let columns = vec![100.0, 200.0, 300.0];
        assert_eq!(find_column_index(&columns, 100.0), Some(0));
        assert_eq!(find_column_index(&columns, 200.0), Some(1));
        assert_eq!(find_column_index(&columns, 300.0), Some(2));
    }

    #[test]
    fn test_find_column_index_closest_within_threshold() {
        let columns = vec![100.0, 200.0, 300.0];
        assert_eq!(find_column_index(&columns, 105.0), Some(0));
        assert_eq!(find_column_index(&columns, 195.0), Some(1));
    }

    #[test]
    fn test_find_column_index_outside_threshold() {
        let columns = vec![100.0, 200.0, 300.0];
        // Threshold is clamped to min 25, max 50 based on min gap / 2
        // Min gap = 100, threshold = clamp(50, 25, 50) = 50
        assert_eq!(find_column_index(&columns, 500.0), None);
    }

    #[test]
    fn test_find_column_index_single_column() {
        let columns = vec![150.0];
        // Single column → threshold = 50.0
        assert_eq!(find_column_index(&columns, 150.0), Some(0));
        assert_eq!(find_column_index(&columns, 170.0), Some(0));
    }

    #[test]
    fn test_find_column_index_empty_columns() {
        let columns: Vec<f32> = vec![];
        assert_eq!(find_column_index(&columns, 100.0), None);
    }

    // --- find_row_index ---

    #[test]
    fn test_find_row_index_exact_match() {
        let rows = vec![500.0, 480.0, 460.0];
        assert_eq!(find_row_index(&rows, 500.0), Some(0));
        assert_eq!(find_row_index(&rows, 480.0), Some(1));
    }

    #[test]
    fn test_find_row_index_within_threshold() {
        let rows = vec![500.0, 480.0, 460.0];
        assert_eq!(find_row_index(&rows, 505.0), Some(0));
        assert_eq!(find_row_index(&rows, 475.0), Some(1));
    }

    #[test]
    fn test_find_row_index_outside_threshold() {
        let rows = vec![500.0, 480.0, 460.0];
        // threshold is 15.0
        assert_eq!(find_row_index(&rows, 400.0), None);
    }

    #[test]
    fn test_find_row_index_single_row() {
        let rows = vec![500.0];
        assert_eq!(find_row_index(&rows, 500.0), Some(0));
        assert_eq!(find_row_index(&rows, 510.0), Some(0));
    }

    // --- find_column_boundaries ---

    #[test]
    fn test_find_column_boundaries_empty() {
        let items: Vec<(usize, &TextItem)> = vec![];
        assert_eq!(
            find_column_boundaries(&items, TableDetectionMode::SmallFont),
            Vec::<f32>::new()
        );
    }

    #[test]
    fn test_find_column_boundaries_two_clusters() {
        // Items at x=100 and x=200 with enough repetition
        let items_data: Vec<TextItem> = (0..10)
            .map(|i| {
                let x = if i % 2 == 0 { 100.0 } else { 200.0 };
                make_item("Cell", x, 500.0 - (i as f32 * 20.0), 10.0)
            })
            .collect();
        let items: Vec<(usize, &TextItem)> = items_data.iter().enumerate().collect();
        let cols = find_column_boundaries(&items, TableDetectionMode::SmallFont);
        assert_eq!(cols.len(), 2);
    }

    #[test]
    fn test_find_column_boundaries_single_item() {
        let item = make_item("Solo", 100.0, 500.0, 10.0);
        let items: Vec<(usize, &TextItem)> = vec![(0, &item)];
        // Single item won't pass the min_items_per_col filter (needs >=2)
        let cols = find_column_boundaries(&items, TableDetectionMode::SmallFont);
        assert!(cols.is_empty());
    }

    #[test]
    fn test_find_column_boundaries_body_font_paragraph_rejection() {
        // All items at same X → >60% in one column → rejected in BodyFont mode
        let items_data: Vec<TextItem> = (0..10)
            .map(|i| make_item("Text", 100.0, 500.0 - (i as f32 * 20.0), 10.0))
            .collect();
        let items: Vec<(usize, &TextItem)> = items_data.iter().enumerate().collect();
        let cols = find_column_boundaries(&items, TableDetectionMode::BodyFont);
        assert!(cols.is_empty());
    }

    #[test]
    fn test_find_column_boundaries_min_items_filter() {
        // Create 10 items at x=100 and 1 item at x=300
        // The single outlier should be filtered out
        let mut items_data: Vec<TextItem> = (0..10)
            .map(|i| make_item("Cell", 100.0, 500.0 - (i as f32 * 20.0), 10.0))
            .collect();
        items_data.push(make_item("Lone", 300.0, 500.0, 10.0));
        let items: Vec<(usize, &TextItem)> = items_data.iter().enumerate().collect();
        let cols = find_column_boundaries(&items, TableDetectionMode::SmallFont);
        // Only the cluster at x=100 should survive
        assert!(cols.len() <= 1);
    }

    // --- find_row_boundaries ---

    #[test]
    fn test_find_row_boundaries_empty() {
        let items: Vec<(usize, &TextItem)> = vec![];
        assert_eq!(find_row_boundaries(&items), Vec::<f32>::new());
    }

    #[test]
    fn test_find_row_boundaries_descending_order() {
        let items_data = vec![
            make_item("A", 100.0, 500.0, 10.0),
            make_item("B", 100.0, 480.0, 10.0),
            make_item("C", 100.0, 460.0, 10.0),
        ];
        let items: Vec<(usize, &TextItem)> = items_data.iter().enumerate().collect();
        let rows = find_row_boundaries(&items);
        assert_eq!(rows.len(), 3);
        // Should be in descending order
        assert!(rows[0] > rows[1]);
        assert!(rows[1] > rows[2]);
    }

    #[test]
    fn test_find_row_boundaries_clustering() {
        // Items close together should cluster into one row
        let items_data = vec![
            make_item("A", 100.0, 500.0, 10.0),
            make_item("B", 200.0, 501.0, 10.0),
            make_item("C", 100.0, 480.0, 10.0),
        ];
        let items: Vec<(usize, &TextItem)> = items_data.iter().enumerate().collect();
        let rows = find_row_boundaries(&items);
        assert_eq!(rows.len(), 2); // 500 and 501 cluster together
    }

    #[test]
    fn test_find_row_boundaries_single_row() {
        let items_data = vec![make_item("A", 100.0, 500.0, 10.0)];
        let items: Vec<(usize, &TextItem)> = items_data.iter().enumerate().collect();
        let rows = find_row_boundaries(&items);
        assert_eq!(rows.len(), 1);
        assert!((rows[0] - 500.0).abs() < 0.01);
    }

    #[test]
    fn test_find_row_boundaries_items_at_same_y() {
        let items_data = vec![
            make_item("A", 100.0, 500.0, 10.0),
            make_item("B", 200.0, 500.0, 10.0),
            make_item("C", 300.0, 500.0, 10.0),
        ];
        let items: Vec<(usize, &TextItem)> = items_data.iter().enumerate().collect();
        let rows = find_row_boundaries(&items);
        assert_eq!(rows.len(), 1);
    }

    // --- recover_header_row ---

    #[test]
    fn test_recover_header_row_prepends_header() {
        let all_items = vec![
            make_item("Col1", 100.0, 520.0, 12.0), // body font, above table
            make_item("Col2", 200.0, 520.0, 12.0), // body font, above table
            make_item("A", 100.0, 500.0, 8.0),     // small font, in table
            make_item("B", 200.0, 500.0, 8.0),
        ];
        let mut table = Table {
            columns: vec![100.0, 200.0],
            rows: vec![500.0, 480.0],
            cells: vec![vec!["A".into(), "B".into()], vec!["C".into(), "D".into()]],
            item_indices: vec![2, 3],
            kind: TableKind::Data,
        };

        recover_header_row(&mut table, &all_items, 9.0);
        assert_eq!(table.cells.len(), 3);
        assert_eq!(table.cells[0], vec!["Col1", "Col2"]);
    }

    #[test]
    fn test_recover_header_row_no_candidates() {
        let all_items = vec![
            make_item("A", 100.0, 500.0, 8.0),
            make_item("B", 200.0, 500.0, 8.0),
        ];
        let mut table = Table {
            columns: vec![100.0, 200.0],
            rows: vec![500.0],
            cells: vec![vec!["A".into(), "B".into()]],
            item_indices: vec![0, 1],
            kind: TableKind::Data,
        };

        let rows_before = table.rows.len();
        recover_header_row(&mut table, &all_items, 9.0);
        assert_eq!(table.rows.len(), rows_before);
    }

    #[test]
    fn test_recover_header_row_skips_strikeout_candidates() {
        let mut old_col1 = make_item("Old Col1", 100.0, 520.0, 12.0);
        old_col1.is_strikeout = true;
        let mut old_col2 = make_item("Old Col2", 200.0, 520.0, 12.0);
        old_col2.is_strikeout = true;
        let all_items = vec![
            old_col1,
            old_col2,
            make_item("A", 100.0, 500.0, 8.0),
            make_item("B", 200.0, 500.0, 8.0),
        ];
        let mut table = Table {
            columns: vec![100.0, 200.0],
            rows: vec![500.0, 480.0],
            cells: vec![vec!["A".into(), "B".into()], vec!["C".into(), "D".into()]],
            item_indices: vec![2, 3],
            kind: TableKind::Data,
        };

        recover_header_row(&mut table, &all_items, 9.0);
        assert_eq!(table.rows.len(), 2);
        assert_eq!(table.cells[0], vec!["A", "B"]);
    }

    #[test]
    fn test_recover_header_row_too_far_above() {
        let all_items = vec![
            make_item("Col1", 100.0, 600.0, 12.0), // way above
            make_item("Col2", 200.0, 600.0, 12.0),
            make_item("A", 100.0, 500.0, 8.0),
            make_item("B", 200.0, 500.0, 8.0),
        ];
        let mut table = Table {
            columns: vec![100.0, 200.0],
            rows: vec![500.0, 480.0],
            cells: vec![vec!["A".into(), "B".into()], vec!["C".into(), "D".into()]],
            item_indices: vec![2, 3],
            kind: TableKind::Data,
        };

        let rows_before = table.rows.len();
        recover_header_row(&mut table, &all_items, 9.0);
        assert_eq!(table.rows.len(), rows_before);
    }

    #[test]
    fn test_recover_header_row_single_column_populated() {
        // Only 1 column populated → not a real header
        let all_items = vec![
            make_item("OnlyCol1", 100.0, 520.0, 12.0),
            make_item("A", 100.0, 500.0, 8.0),
            make_item("B", 200.0, 500.0, 8.0),
        ];
        let mut table = Table {
            columns: vec![100.0, 200.0],
            rows: vec![500.0],
            cells: vec![vec!["A".into(), "B".into()]],
            item_indices: vec![1, 2],
            kind: TableKind::Data,
        };

        let rows_before = table.rows.len();
        recover_header_row(&mut table, &all_items, 9.0);
        assert_eq!(table.rows.len(), rows_before);
    }

    #[test]
    fn test_recover_header_row_empty_table() {
        let all_items = vec![make_item("Col1", 100.0, 520.0, 12.0)];
        let mut table = Table {
            columns: vec![],
            rows: vec![],
            cells: vec![],
            item_indices: vec![],
            kind: TableKind::Data,
        };

        recover_header_row(&mut table, &all_items, 9.0);
        assert!(table.cells.is_empty());
    }

    #[test]
    fn test_find_column_boundaries_dense_schedule() {
        // Simulate a 24-column train schedule with ~26pt column spacing and
        // per-glyph items that create many X-positions within each column.
        let mut items: Vec<(usize, TextItem)> = Vec::new();
        let mut rng_offset = 0.0f32;
        for col in 0..24 {
            let base_x = 50.0 + col as f32 * 26.0;
            // ~50 items per column with ±2pt jitter to simulate per-glyph text
            for row in 0..50 {
                rng_offset = (rng_offset + 0.7) % 4.0; // deterministic pseudo-jitter
                let x = base_x + rng_offset - 2.0;
                let y = 700.0 - row as f32 * 12.0;
                items.push((
                    0,
                    TextItem {
                        text: format!("{}", row),
                        x,
                        y,
                        width: 8.0,
                        font_size: 7.0,
                        height: 7.0,
                        font: String::new(),
                        font_tag: String::new(),
                        legacy_symbol_rewrite: false,
                        is_bold: false,
                        is_italic: false,
                        is_underline: false,
                        is_strikeout: false,
                        rotation: 0.0,
                        advance_known: true,
                        item_type: ItemType::Text,
                        mcid: None,
                        baseline_shift: 0.0,
                        page: 1,
                    },
                ));
            }
        }
        let refs: Vec<(usize, &TextItem)> = items.iter().map(|(i, t)| (*i, t)).collect();
        let cols = find_column_boundaries(&refs, TableDetectionMode::SmallFont);
        // Should find close to 24 columns (within ±2)
        assert!(
            cols.len() >= 22 && cols.len() <= 26,
            "Expected ~24 columns, got {}",
            cols.len()
        );
    }

    #[test]
    fn test_find_column_boundaries_wide_spacing_still_works() {
        // Normal table with 4 widely-spaced columns — should still work
        let mut items = Vec::new();
        for col in 0..4 {
            let base_x = 50.0 + col as f32 * 120.0;
            for row in 0..10 {
                items.push((
                    0,
                    TextItem {
                        text: format!("cell_{}_{}", col, row),
                        x: base_x + (row as f32 * 0.3),
                        y: 700.0 - row as f32 * 15.0,
                        width: 40.0,
                        font_size: 10.0,
                        height: 7.0,
                        font: String::new(),
                        font_tag: String::new(),
                        legacy_symbol_rewrite: false,
                        is_bold: false,
                        is_italic: false,
                        is_underline: false,
                        is_strikeout: false,
                        rotation: 0.0,
                        advance_known: true,
                        item_type: ItemType::Text,
                        mcid: None,
                        baseline_shift: 0.0,
                        page: 1,
                    },
                ));
            }
        }
        let refs: Vec<(usize, &TextItem)> = items.iter().map(|(i, t)| (*i, t)).collect();
        let cols = find_column_boundaries(&refs, TableDetectionMode::BodyFont);
        assert_eq!(cols.len(), 4, "Expected 4 columns, got {}", cols.len());
    }
}
