# Test Files Directory

This directory is used to store test media files for the integration tests.

## Adding Test Files

To run the integration tests with real media files, you should add some test media files to this directory. The tests will look for files with the following extensions:

- Video files: `.mp4`, `.mkv`
- Audio files: `.mp3`, `.wav`

## Test File Requirements

- The files should be small (a few seconds long) to keep the tests running quickly.
- The files should be valid media files that can be processed by ffmpeg.
- Include at least one video file and one audio file for comprehensive testing.

## Example Test Files

You can create simple test files using ffmpeg:

```bash
# Create a 3-second test video file
ffmpeg -f lavfi -i testsrc=duration=3:size=320x240:rate=30 -c:v libx264 test_video.mp4

# Create a 3-second test audio file
ffmpeg -f lavfi -i sine=frequency=1000:duration=3 -c:a libmp3lame test_audio.mp3
```

## Note

If no test files are found in this directory, the integration tests will create dummy files, but these dummy files won't work with actual ffmpeg processing. The tests will be skipped in this case.

The unit tests will still run without real test files, as they use mocks to simulate the behavior of ffmpeg.
