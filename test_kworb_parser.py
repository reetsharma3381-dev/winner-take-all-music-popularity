SAMPLE_HTML = """
<html><body>
<table>
<tr><th>Pos</th><th>P+</th><th>Artist and Title</th><th>Days</th><th>Pk</th><th>(x?)</th><th>Streams</th><th>Streams+</th><th>7Day</th><th>7Day+</th><th>Total</th></tr>
<tr>
<td>1</td><td>=</td>
<td><a href="https://kworb.net/spotify/artist/1McMsnEElThX1knmY4oliG.html">Olivia Rodrigo</a> - <a href="https://kworb.net/spotify/track/49j6SvuvWfbEKZKzsHCdLJ.html">stupid song</a></td>
<td>7</td><td>1</td><td>(x7)</td><td>6,250,964</td><td>-359,994</td><td>50,369,677</td><td>+6,250,964</td><td>50,369,677</td>
</tr>
<tr>
<td>6</td><td>+1</td>
<td><a href="https://kworb.net/spotify/artist/1uNFoZAHBGtllmzznpCI3s.html">Justin Bieber</a> - <a href="https://kworb.net/spotify/track/6QFCMUUq1T2Vf5sFUXcuQ7.html">Beauty And A Beat</a> (w/ <a href="https://kworb.net/spotify/artist/0hCNtLu0JehylgoiP8L4Gh.html">Nicki Minaj</a>)</td>
<td>112</td><td>1</td><td>(x31)</td><td>4,565,877</td><td>+151,135</td><td>31,357,560</td><td>+114,218</td><td>485,549,137</td>
</tr>
<tr>
<td>35</td><td>+1</td>
<td><a href="https://kworb.net/spotify/artist/06HL4z0CvFAxyc27GXpf02.html">Taylor Swift</a> - <a href="https://kworb.net/spotify/track/5uPaqMMt59KGrdKIitDRqa.html">I Knew It, I Knew You</a></td>
<td>14</td><td>1</td><td>(x1)</td><td>2,626,143</td><td>+53,435</td><td>19,081,020</td><td>-849,997</td><td>50,158,029</td>
</tr>
<tr>
<td>37</td><td>=</td>
<td><a href="https://kworb.net/spotify/artist/06HL4z0CvFAxyc27GXpf02.html">Taylor Swift</a> - <a href="https://kworb.net/spotify/track/53iuhJlwXhSER5J2IYYv1W.html">The Fate of Ophelia</a></td>
<td>259</td><td>1</td><td>(x78)</td><td>2,596,276</td><td>+63,131</td><td>17,539,219</td><td>-36,448</td><td>1,434,330,792</td>
</tr>
</table>
</body></html>
"""

if __name__ == "__main__":
    from collect_kworb_wayback_data import parse_chart_page
    import pandas as pd

    records = parse_chart_page(SAMPLE_HTML)
    df = pd.DataFrame(records)
    print(df)
    print()

    # Check: Taylor Swift's two songs should be summable to one artist total
    grouped = df.groupby(["artist_id", "artist_name"], as_index=False)["streams_est"].sum()
    print(grouped)

    assert len(records) == 5, f"expected 5 artist-credit rows, got {len(records)}"  # Bieber row has 2 artists
    taylor_total = grouped.loc[grouped["artist_name"] == "Taylor Swift", "streams_est"].iloc[0]
    assert taylor_total == 2_626_143 + 2_596_276, f"Taylor Swift sum wrong: {taylor_total}"
    print("\nAll checks passed.")
