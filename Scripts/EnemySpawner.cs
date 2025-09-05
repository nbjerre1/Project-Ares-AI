using Godot;
using System;

public partial class EnemySpawner : Node2D
{
    [Export] public PackedScene EnemyScene { get; set; }
    [Export] public float SpawnInterval { get; set; } = 2.0f;
    [Export] public int MaxEnemies { get; set; } = 5;
    [Export] public NodePath CameraPath { get; set; } // Assign your Camera2D node in the editor
    [Export] public float SpawnDistance { get; set; } = 300f; // Distance from camera edge

    private float _timer = 0f;
    private int _currentEnemies = 0;
    private Camera2D _camera;

    public override void _Ready()
    {
        if (CameraPath != null)
            _camera = GetNode<Camera2D>(CameraPath);

        _timer = SpawnInterval; // This will trigger the first spawn immediately

        // If you need to reference the player, use GetParent().GetNode<Player>("Player") or similar,
        // to ensure you are referencing the local scene's player, not a global singleton.
    }

    public override void _Process(double delta)
    {
        if (EnemyScene == null || _currentEnemies >= MaxEnemies)
            return;

        _timer += (float)delta;
        if (_timer >= SpawnInterval)
        {
            SpawnEnemy();
            _timer = 0f;
        }
    }

    private void SpawnEnemy()
    {
        if (_camera == null)
            return;

        var enemy = EnemyScene.Instantiate() as Node2D;
        if (enemy != null)
        {
            enemy.Position = GetSpawnPositionOutsideCamera();
            GetParent().AddChild(enemy);
            _currentEnemies++;

            if (enemy is Mushroom mushroom)
            {
                mushroom.GetNode<HP>("Health").Died += OnEnemyDied;
            }
        }
    }

    private Vector2 GetSpawnPositionOutsideCamera()
    {
        // Get camera position and viewport size
        Vector2 camPos = _camera.GlobalPosition;
        Vector2 viewportSize = GetViewport().GetVisibleRect().Size;

        // Calculate camera bounds
        float left = camPos.X - viewportSize.X / 2;
        float right = camPos.X + viewportSize.X / 2;
        float top = camPos.Y - viewportSize.Y / 2;
        float bottom = camPos.Y + viewportSize.Y / 2;

        // Randomly pick a side to spawn on
        int side = (int)(GD.Randi() % 4);
        float x = 0, y = 0;

        switch (side)
        {
            case 0: // Left
                x = left - SpawnDistance;
                y = GD.Randf() * (bottom - top) + top;
                break;
            case 1: // Right
                x = right + SpawnDistance;
                y = GD.Randf() * (bottom - top) + top;
                break;
            case 2: // Top
                x = GD.Randf() * (right - left) + left;
                y = top - SpawnDistance;
                break;
            case 3: // Bottom
                x = GD.Randf() * (right - left) + left;
                y = bottom + SpawnDistance;
                break;
        }

        return new Vector2(x, y);
    }

    private void OnEnemyDied()
    {
        _currentEnemies--;
    }
}
